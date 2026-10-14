"""Neural network modules for OGPO.

This package contains organized neural network components including:
- MLP variants (MLP, MLPCond, MLPWithFiLM, SimBaMLP)
- Attention and transformer components
- Distribution wrappers
- Miscellaneous utilities
"""

# MLP components
from vadar.networks.modules.mlp import (
    MLP,
    MLPCond,
    MLPWithFiLM,
    MLPResidualBlock,
    SimBaMLP,
    SimBaInternalMLP,
    RSNorm,
    default_init,
)

# Attention and transformer components
from vadar.networks.modules.attention import (
    MultiHeadAttention,
    FeedForward,
    AdaLNLayer,
    AdaLNFinalLayer,
    AdaLNTransformer,
    CrossAttention,
    CrossAttnLayer,
    CrossAttnFinalLayer,
    TimestepEmbedder,
    ConditioningMLP,
    modulate,
)

# Distribution components
from vadar.networks.modules.distributions import (
    TanhMultivariateNormalDiag,
    TransformedWithMode,
    RescaleFromTanhBijector,
)

# Time embedding
from vadar.networks.modules.time_embedding import (
    SinusoidalTimeEmbedding,
)

# Miscellaneous utilities
from vadar.networks.modules.misc import (
    NoiseInjectionNetwork,
    LogParam,
    Identity,
    orthogonal_init,
    zeros_init,
)

# Two-tier observation encoder (image + proprio fusion)
from vadar.networks.modules.two_tier import TwoTierObsEncoder

__all__ = [
    # MLP components
    'MLP',
    'MLPCond',
    'MLPWithFiLM',
    'MLPResidualBlock',
    'SimBaMLP',
    'SimBaInternalMLP',
    'RSNorm',
    'default_init',
    # Attention components
    'MultiHeadAttention',
    'FeedForward',
    'AdaLNLayer',
    'AdaLNFinalLayer',
    'AdaLNTransformer',
    'CrossAttention',
    'CrossAttnLayer',
    'CrossAttnFinalLayer',
    'TimestepEmbedder',
    'ConditioningMLP',
    'modulate',
    # Distribution components
    'TanhMultivariateNormalDiag',
    'TransformedWithMode',
    'RescaleFromTanhBijector',
    # Time embedding
    'SinusoidalTimeEmbedding',
    # Misc utilities
    'NoiseInjectionNetwork',
    'LogParam',
    'Identity',
    'orthogonal_init',
    'zeros_init',
    # Two-tier obs encoder
    'TwoTierObsEncoder',
]
