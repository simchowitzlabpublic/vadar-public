"""VaDaR critics: one value per subgoal, returned with shape (G, M, B).

The three parameterizations are drop-in replacements for each other and share
the call signature of `Value`, so the VaDaR agents never need to know which
one they were given.

- VadarCritic (default, the paper's "VaDaR critic"): a goal-conditioned critic
  Q(s, a, g). One shared network is queried once per subgoal g, with g given as
  a learned embedding of the one-hot subgoal index (generated internally, so
  observations need no goal suffix).
- VectorVadarCritic ("Vector-VaDaR"): a vector-valued critic S x A -> R^G, i.e.
  a shared backbone with G output heads evaluated in one forward pass.
- SeparateVadarCritic: G independent critics with no shared parameters
  (the paper's "separate critics" ablation).

MIQ, the iterative critic, is `ValueMIPEnsemble(num_subgoals=G)`.
"""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from vadar.networks.critics import Value


class VadarCritic(nn.Module):
    """Goal-conditioned critic Q(s, a, g), evaluated at every subgoal g."""
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    num_subgoals: int = 1
    goal_embed_dim: int = 32

    def setup(self):
        self.value_net = Value(
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
            num_ensembles=self.num_ensembles,
        )
        self.goal_proj = nn.Dense(self.goal_embed_dim)

    def __call__(self, observations, actions=None, is_encoded=False, **kwargs):
        G = self.num_subgoals
        goal_embeds = nn.gelu(self.goal_proj(jnp.eye(G, dtype=jnp.float32)))  # (G, goal_embed_dim)
        qs = []
        for g in range(G):
            embed = jnp.broadcast_to(goal_embeds[g], (*observations.shape[:-1], self.goal_embed_dim))
            obs_g = jnp.concatenate([observations, embed], axis=-1)
            qs.append(self.value_net(obs_g, actions=actions, is_encoded=True))  # (M, B)
        return jnp.stack(qs, axis=0)  # (G, M, B)


class VectorVadarCritic(nn.Module):
    """Vector-valued critic S x A -> R^G with a shared backbone and G heads."""
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    num_subgoals: int = 1

    def setup(self):
        self.value_net = Value(
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
            num_ensembles=self.num_ensembles,
            num_subgoals=self.num_subgoals,
        )

    def __call__(self, observations, actions=None, is_encoded=False, **kwargs):
        q = self.value_net(observations, actions=actions, is_encoded=True)
        return q if self.num_subgoals > 1 else q[None]  # (G, M, B)


class SeparateVadarCritic(nn.Module):
    """G independent critics, one per subgoal (no representation sharing)."""
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    num_subgoals: int = 1

    def setup(self):
        per_goal = nn.vmap(
            Value,
            variable_axes={'params': 0, 'intermediates': 0},
            split_rngs={'params': True, 'dropout': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_subgoals,
        )
        self.value_net = per_goal(
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
            num_ensembles=self.num_ensembles,
        )

    def __call__(self, observations, actions=None, is_encoded=False, **kwargs):
        return self.value_net(observations, actions, False, True)  # (G, M, B)


VADAR_CRITICS = {
    'goal_conditioned': VadarCritic,
    'vector': VectorVadarCritic,
    'separate': SeparateVadarCritic,
}
