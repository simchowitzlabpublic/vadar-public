"""OGPO Networks Module.

Neural network architectures for reinforcement learning including:
- Actors: Policy networks (flow matching, Gaussian, edit policies)
- Critics: Q-function and value networks
- Encoders: Vision encoders (IMPALA, MinViT)
- Modules: Building blocks (MLP, attention, distributions)

All networks are implemented in JAX/Flax.
"""

# Actor networks
from vadar.networks.actors import (
    Actor,
    ActorVectorField,
    ActorVectorFieldTF,
    ActorVectorFieldSimBa,
    EditPolicy,
    EditActor,
)

# FQL-specific networks
from vadar.networks.actors_fql import (
    OneStepPolicy,
)

# Critic networks
from vadar.networks.critics import (
    Value,
    ValueTF,
    ValueSimBa,
    ValueMIP,
    ValueMIPEnsemble,
    ensemblize,
)

# Miscellaneous utilities
from vadar.networks.modules.misc import (
    NoiseInjectionNetwork,
    LogParam,
    Identity,
    orthogonal_init,
    zeros_init,
)

# Encoder registries
from vadar.networks.encoders import (
    encoder_modules,
    ImpalaEncoder,
    MinViT,
    MinVitEncoder,
)

__all__ = [
    # Actors
    'Actor',
    'ActorVectorField',
    'ActorVectorFieldTF',
    'ActorVectorFieldSimBa',
    'EditPolicy',
    'EditActor',
    'OneStepPolicy',  # FQL
    # Critics
    'Value',
    'ValueTF',
    'ValueSimBa',
    'ValueMIP',
    'ValueMIPEnsemble',
    'ensemblize',
    # Utilities
    'NoiseInjectionNetwork',
    'LogParam',
    'Identity',
    'orthogonal_init',
    'zeros_init',
    # Encoders
    'encoder_modules',
    'ImpalaEncoder',
    'MinViT',
    'MinVitEncoder',
]
