"""Per-layer (instead of whole-backbone) gradient checkpointing.

``custom_model.py``'s ``gradient_checkpointing`` flag used to wrap the entire
backbone forward pass (encoder + transformer tower + decoder) in a single
``hk.remat``. That's enough to keep --mode linear-probe (frozen, detached
backbone) within memory, but for --mode lora -- where gradients must flow
through the whole, un-detached backbone -- a single coarse-grained remat
still needed ~55GB even after XLA's own rematerialization pass. Switching to
this module's per-layer remat only trimmed that to ~53GB (confirmed
empirically: 27553692/27775091 OOM'd trying to allocate that much on an
~80GB GPU either way) -- a real but modest reduction, *not* by itself enough
to fit. The fix that actually closed the gap was in
``finetune/train.py::grad_step``: applying ``jax.lax.stop_gradient`` to each
individual frozen backbone weight before the forward pass, so autodiff never
computes (and doesn't need memory for) a real backward pass through those
~450M frozen parameters at all -- confirmed empirically too: the first run
that combined both fixes compiled a graph with zero ``convBackwardFilter``
ops for the backbone (versus hundreds before) and trained without OOM.

Keep this module anyway: per-layer remat still reduces peak memory for
whatever *does* need a real backward pass (the LoRA-adapted q/v projections
and everything downstream of them, plus a full-model backward on
--mode full, if that's ever added), matching alphagenome-pytorch's own
per-layer ``torch.utils.checkpoint`` granularity rather than checkpointing
the whole backbone as one opaque unit.

This module monkeypatches alphagenome_research.model.model's SequenceEncoder,
TransformerTower and SequenceDecoder to apply ``hk.remat`` around each
individual block/layer instead. Unlike alphagenome_ft.lora's backbone patch,
this is always safe to install eagerly (including before dna_model.create()'s
own internal checkpoint restore): hk.remat only wraps *how* a computation
graph is executed, it doesn't add, rename, or reshape any parameter, so it
never changes the restore target's structure the way adding new LoRA
parameters does.
"""

from __future__ import annotations

import haiku as hk

_PATCHED = False


def install_fine_grained_remat() -> None:
    """Monkeypatch the backbone's encoder/tower/decoder for per-block remat.

    Idempotent: safe to call multiple times (e.g. once per CLI invocation).
    """
    global _PATCHED
    if _PATCHED:
        return

    import haiku._src.module as hk_module_internal
    from alphagenome_research.model import attention as attention_module
    from alphagenome_research.model import convolutions
    from alphagenome_research.model import layers
    from alphagenome_research.model import model as model_module

    def _wrap(cls, method_name, fn):
        setattr(
            cls, method_name,
            hk_module_internal.wrap_method(method_name, fn, lambda: cls),
        )

    def _encoder_call(self, dna_sequence, *, is_training):
        intermediates = {}
        x = convolutions.DnaEmbedder()(dna_sequence, is_training=is_training)
        intermediates['bin_size_1'] = x
        x = layers.pool(x)
        for block_idx, bin_size in enumerate([2, 4, 8, 16, 32, 64]):
            def _step(x, block_idx=block_idx):
                return convolutions.DownResBlock(f'downres_block_{block_idx}')(
                    x, is_training=is_training
                )
            x = hk.remat(_step)(x)
            intermediates[f'bin_size_{bin_size}'] = x
            x = layers.pool(x)
        return x, intermediates

    def _decoder_call(self, x, intermediates, *, is_training):
        for bin_size in [64, 32, 16, 8, 4, 2, 1]:
            def _step(x, bin_size=bin_size):
                return convolutions.UpResBlock()(
                    x, intermediates[f'bin_size_{bin_size}'], is_training=is_training
                )
            x = hk.remat(_step)(x)
        return x

    def _tower_call(self, x, *, is_training):
        pair_x = None
        for i in range(9):
            def _step(x, pair_x, i=i):
                local_pair_x = pair_x
                if i % 2 == 0:
                    local_pair_x = attention_module.PairUpdateBlock()(x, local_pair_x)
                mha_bias = attention_module.AttentionBiasBlock()(
                    local_pair_x, is_training
                )
                x = x + attention_module.MHABlock()(
                    x, mha_bias, is_training=is_training
                )
                x = x + attention_module.MLPBlock()(x, is_training=is_training)
                return x, local_pair_x
            x, pair_x = hk.remat(_step)(x, pair_x)
        return x, pair_x

    _wrap(model_module.SequenceEncoder, '__call__', _encoder_call)
    _wrap(model_module.SequenceDecoder, '__call__', _decoder_call)
    _wrap(model_module.TransformerTower, '__call__', _tower_call)

    _PATCHED = True
