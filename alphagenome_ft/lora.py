"""
LoRA (Low-Rank Adaptation) utilities for AlphaGenome finetuning.

Provides building blocks for parameter-efficient finetuning by adding small
low-rank matrices to selected linear layers while keeping the backbone frozen.

Typical usage pattern:
1. Write custom Haiku modules/heads that use LoRALinear instead of hk.Linear.
2. Pass frozen backbone embeddings as input to those modules.
3. Freeze backbone parameters with parameter_utils.freeze_except_lora.
4. Train only the LoRA parameters (lora_a, lora_b).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from alphagenome_ft.parameter_utils import _keypath_to_str

# Matches PyTorch's default nn.Linear(in, rank) init used for LoRA's "A" matrix
# (alphagenome_pytorch.extensions.finetuning.adapters.LoRA calls
# nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)), which is just
# nn.Linear's own reset_parameters default). For a Linear(in, rank) weight,
# kaiming_uniform_ with a=sqrt(5) gives gain=sqrt(2/(1+5))=sqrt(1/3) and bound
# = gain*sqrt(3/fan_in) = sqrt(1/fan_in), i.e. U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
# Haiku's VarianceScaling(scale, mode='fan_in', distribution='uniform') uses
# limit=sqrt(3*scale/fan_in); solving sqrt(3*scale/fan_in) == 1/sqrt(fan_in)
# gives scale=1/3.
_LORA_A_INIT = hk.initializers.VarianceScaling(
    scale=1.0 / 3.0, mode='fan_in', distribution='uniform',
)
_LORA_B_INIT = hk.initializers.Constant(0.0)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters.

    Attributes:
        rank: Rank of the low-rank decomposition. Lower values mean fewer
            trainable parameters. Common values: 4, 8, 16.
        alpha: Scaling factor applied to the LoRA output. The effective
            scale is ``alpha / rank``. Setting alpha == rank gives scale 1.
    """
    rank: int = 8
    alpha: float = 1.0


class LoRALinear(hk.Module):
    """Linear layer augmented with a trainable low-rank adapter.

    Forward pass computes x @ W + (x @ A) @ B * (alpha / rank)
    where ``W`` is a base weight (typically kept frozen by the caller via
    ``jax.lax.stop_gradient`` or ``parameter_utils.freeze_backbone_keep_lora``)
    and ``A``, ``B`` are small trainable matrices.

    The parameter ``W`` is stored under the Haiku key ``"w"``, the adapters
    under ``"lora_a"`` and ``"lora_b"``.  This mirrors the naming used by
    AlphaGenome's own linear layers so that LoRA modules placed inside a
    backbone-matching name scope will coexist cleanly with loaded checkpoints.

    Args:
        out_dim: Output feature dimension.
        config: LoRA hyperparameters (rank and alpha). Defaults to
            ``LoRAConfig()`` (rank=8, alpha=1.0).
        with_bias: Whether to add a bias term to the base linear.
        name: Optional Haiku module name.
    """
    def __init__(
        self,
        out_dim: int,
        config: LoRAConfig | None = None,
        with_bias: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.out_dim = out_dim
        self.config = config or LoRAConfig()
        self.with_bias = with_bias

    def __call__(self, x: Float[Array, '... D_in']) -> Float[Array, '... D_out']:
        in_dim = x.shape[-1]
        rank = self.config.rank
        alpha = self.config.alpha

        w = hk.get_parameter(
            'w',
            shape=(in_dim, self.out_dim),
            dtype=x.dtype,
            init=hk.initializers.VarianceScaling(),
        )
        a = hk.get_parameter(
            'lora_a',
            shape=(in_dim, rank),
            dtype=x.dtype,
            init=hk.initializers.RandomNormal(stddev=0.01),
        )
        b = hk.get_parameter(
            'lora_b',
            shape=(rank, self.out_dim),
            dtype=x.dtype,
            init=hk.initializers.Constant(0.0),
        )

        base = x @ w
        if self.with_bias:
            bias = hk.get_parameter(
                'b',
                shape=(self.out_dim,),
                dtype=x.dtype,
                init=hk.initializers.Constant(0.0),
            )
            base = base + bias

        delta = (x @ a) @ b * (alpha / rank)
        return base + delta


def get_lora_parameter_paths(params: PyTree) -> list[str]:
    """Return all parameter paths that correspond to LoRA adapter matrices.

    A path is considered a LoRA path when its final segment is ``lora_a`` or
    ``lora_b``.

    Args:
        params: Haiku parameter tree (e.g. ``model._params``).

    Returns:
        List of slash-delimited path strings, e.g.
        ``['my_head/lora_linear/lora_a', 'my_head/lora_linear/lora_b']``.
    """
    paths: list[str] = []

    def collect(path_tuple, value):
        if not hasattr(value, 'shape'):
            return
        path_str = _keypath_to_str(path_tuple)
        leaf = path_str.split('/')[-1]
        if leaf in ('lora_a', 'lora_b'):
            paths.append(path_str)

    jax.tree_util.tree_map_with_path(collect, params)
    return paths


_PYTORCH_TO_JAX_TARGET = {
    # alphagenome-pytorch's MHABlock (attention.py) names its q/k/v Linear
    # submodules q_proj/k_proj/v_proj; alphagenome_research's MHABlock
    # (explicitly "Matches JAX: alphagenome_research.model.attention.MHABlock"
    # per the PyTorch source) names the equivalent inline hk.Linear calls
    # q_layer/k_layer/v_layer. Translate so --lora-targets can use the same
    # substrings as the PyTorch reference's default ("q_proj,v_proj").
    'q_proj': 'q_layer',
    'k_proj': 'k_layer',
    'v_proj': 'v_layer',
}


@dataclass
class BackboneLoRAConfig:
    """Which backbone attention projections get a LoRA adapter, and how big.

    Mirrors alphagenome-pytorch's ``apply_lora(model, lora_targets, rank,
    alpha)`` defaults (rank=8, alpha=16, targets=['q_proj', 'v_proj']) — but
    scoped to the JAX/Haiku equivalent names via ``_PYTORCH_TO_JAX_TARGET``.
    """
    rank: int = 8
    alpha: float = 16.0
    targets: tuple[str, ...] = ('q_layer', 'v_layer')

    @classmethod
    def from_pytorch_style_targets(
        cls, targets: Sequence[str], *, rank: int = 8, alpha: float = 16.0,
    ) -> 'BackboneLoRAConfig':
        jax_targets = tuple(
            _PYTORCH_TO_JAX_TARGET.get(t.strip(), t.strip()) for t in targets if t.strip()
        )
        unknown = set(jax_targets) - {'q_layer', 'k_layer', 'v_layer'}
        if unknown:
            raise ValueError(
                f"Unknown LoRA backbone target(s) {sorted(unknown)!r}; "
                "MHABlock only exposes q_layer, k_layer, v_layer "
                "(pytorch-style: q_proj, k_proj, v_proj)."
            )
        return cls(rank=rank, alpha=alpha, targets=jax_targets)


_ORIGINAL_MHABLOCK_CALL = None


def install_mha_backbone_lora(config: BackboneLoRAConfig) -> None:
    """Monkeypatch ``alphagenome_research.model.attention.MHABlock.__call__``
    to add LoRA adapters on the targeted q/k/v projections.

    Haiku has no PyTorch-style ``setattr(parent, name, wrapped_module)``
    mechanism to swap a submodule after construction — MHABlock builds its
    q/k/v projections as inline, name-scoped ``hk.Linear(..., name='q_layer')``
    calls inside ``__call__``, not persistent attributes. Replacing the class's
    ``__call__`` (looked up dynamically by every caller, e.g. TransformerTower)
    is the practical equivalent: every subsequent construction/call picks up
    the patched version automatically, with zero changes needed elsewhere.

    This patches the class **in this process only** — it has no effect on any
    other already-running or future process that happens to import the same
    installed package (e.g. a concurrently running linear-probe job).

    The patched version keeps the exact same ``name='q_layer'``/``'v_layer'``
    for the base projection, so a pretrained checkpoint's weights still load
    into it unchanged; new ``lora_a``/``lora_b`` parameters are added as
    siblings under a small nested name scope (``q_lora``/``v_lora``) so
    :func:`get_lora_parameter_paths` finds them without modification.

    Call this once, before constructing the model, when running in LoRA mode.
    Idempotent: calling it again just re-applies the same patch.
    """
    global _ORIGINAL_MHABLOCK_CALL
    from alphagenome_research.model import attention as attention_module
    from alphagenome_research.model import layers

    if _ORIGINAL_MHABLOCK_CALL is None:
        _ORIGINAL_MHABLOCK_CALL = attention_module.MHABlock.__call__

    rank, alpha = config.rank, config.alpha
    apply_q_lora = 'q_layer' in config.targets
    apply_k_lora = 'k_layer' in config.targets
    apply_v_lora = 'v_layer' in config.targets

    def _lora_linear(h, out_dim, *, name, scope_name):
        base = hk.Linear(out_dim, with_bias=False, name=name)(h)
        with hk.experimental.name_scope(scope_name):
            a = hk.get_parameter(
                'lora_a', shape=(h.shape[-1], rank), dtype=h.dtype, init=_LORA_A_INIT,
            )
            b = hk.get_parameter(
                'lora_b', shape=(rank, out_dim), dtype=h.dtype, init=_LORA_B_INIT,
            )
        delta = (h @ a) @ b * (alpha / rank)
        return base + delta

    def _patched_call(self, x, attention_bias, *, is_training):
        batch_size, seq_len, _ = x.shape
        h = layers.RMSBatchNorm()(x, is_training=is_training)

        if apply_q_lora:
            q_pre = _lora_linear(h, 8 * 128, name='q_layer', scope_name='q_lora')
        else:
            q_pre = hk.Linear(8 * 128, with_bias=False, name='q_layer')(h)
        q = layers.LayerNorm(name='norm_q')(
            q_pre.reshape(batch_size, seq_len, 8, 128)
        )

        if apply_k_lora:
            k_pre = _lora_linear(h, 128, name='k_layer', scope_name='k_lora')
        else:
            k_pre = hk.Linear(128, with_bias=False, name='k_layer')(h)
        k = layers.LayerNorm(name='norm_k')(
            k_pre.reshape(batch_size, seq_len, 1, 128)
        )

        if apply_v_lora:
            v_pre = _lora_linear(h, 192, name='v_layer', scope_name='v_lora')
        else:
            v_pre = hk.Linear(192, with_bias=False, name='v_layer')(h)
        v = layers.LayerNorm(name='norm_v')(
            v_pre.reshape(batch_size, seq_len, 1, 192)
        )

        q = attention_module.apply_rope(q, None, max_position=attention_module._MAX_RELATIVE_DISTANCE)
        k = attention_module.apply_rope(k, None, max_position=attention_module._MAX_RELATIVE_DISTANCE)

        logits_dtype = jnp.float32
        attention_logits = jnp.einsum(
            'bshc,bS1c->bhsS',
            q,
            k,
            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
            preferred_element_type=logits_dtype,
        )
        attention_logits = attention_logits / math.sqrt(128.0)
        attention_logits = (attention_logits + attention_bias).astype(logits_dtype)
        logits_soft_cap = 5.0
        attention_logits = (
            jnp.tanh(attention_logits / logits_soft_cap) * logits_soft_cap
        )
        attention_weights = jax.nn.softmax(attention_logits, axis=-1)

        y = jnp.einsum(
            'bhsS,bS1c->bshc',
            attention_weights,
            v,
            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
        ).astype(q.dtype)
        y = hk.Linear(
            x.shape[-1],
            name='linear_embedding',
            w_init=hk.initializers.TruncatedNormal(stddev=1e-6),
        )(y.reshape(batch_size, seq_len, -1))
        return layers.RMSBatchNorm()(y, is_training=is_training)

    # Directly assigning `_patched_call` to the class would bypass Haiku's own
    # method-wrapping (applied once, at class-definition time, to push/pop the
    # module's name onto Haiku's internal name-scope stack) -- every
    # parameter created inside would then land at the top level instead of
    # nested under this MHABlock instance's scope (e.g. bare 'q_layer'
    # instead of '.../mha_block/q_layer'), breaking pretrained-checkpoint
    # restore. Re-apply the same wrapper Haiku itself uses so the patched
    # method still enters the module's name scope correctly.
    import haiku._src.module as hk_module_internal

    attention_module.MHABlock.__call__ = hk_module_internal.wrap_method(
        '__call__', _patched_call, lambda: attention_module.MHABlock,
    )


def uninstall_mha_backbone_lora() -> None:
    """Restore the original (unpatched) ``MHABlock.__call__``, if patched."""
    global _ORIGINAL_MHABLOCK_CALL
    if _ORIGINAL_MHABLOCK_CALL is not None:
        from alphagenome_research.model import attention as attention_module
        attention_module.MHABlock.__call__ = _ORIGINAL_MHABLOCK_CALL
        _ORIGINAL_MHABLOCK_CALL = None


def count_lora_parameters(params: PyTree) -> int:
    """Count the total number of trainable LoRA adapter elements.

    Args:
        params: Haiku parameter tree (e.g. ``model._params``).

    Returns:
        Total element count across all ``lora_a`` and ``lora_b`` arrays.
    """
    lora_paths = set(get_lora_parameter_paths(params))
    total = 0

    def accumulate(path_tuple, value):
        nonlocal total
        if not hasattr(value, 'size'):
            return
        if _keypath_to_str(path_tuple) in lora_paths:
            total += value.size

    jax.tree_util.tree_map_with_path(accumulate, params)
    return total
