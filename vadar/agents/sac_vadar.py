"""SAC + VaDaR (Value Disaggregation): a copy of sac.py with only the VaDaR changes.

Ported from the QIPM fork (ogpo/agents/sac_qipm.py). Relative to sac.py:
1. the critic returns (G, M, B), one value per subgoal (VADAR_CRITICS, or
   ValueMIPEnsemble(num_subgoals=G) for MIQ);
2. the TD target is compute_vadar_td_target on batch['subgoal_dones'][:, -1, :]
   (per-subgoal discounts with use_subgoal_discounts; 'cost' heads via
   subgoal_head_types; with entropy_backup, temp * log_pi' is subtracted from
   every subgoal's next value);
3. the actor maximizes Q_agg = sum_g w(d)[g] Q(s, a, g), with
   w = subgoal_weights(dones, weighting, subgoal_dag, subgoal_weights).

The PushT multi-objective "avoid obstacle" agent of the fork is this agent with
subgoal_head_types=['subgoal', 'cost'], weighting='constant',
subgoal_weights=[1.0, safety_coeff] (and vadar_critic='vector' for the fork's
shared-backbone critic).
"""
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from vadar.networks.encoders import encoder_modules
from vadar.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from vadar.networks import Actor, Value, LogParam
from vadar.networks.goal_net import GoalConditionedNetwork
from vadar.networks.vadar_critics import VADAR_CRITICS
from vadar.agents.modules.vadar_helper import (
    aggregate_subgoal_values,
    compute_vadar_td_target,
    subgoal_discount_factors,
    subgoal_weights,
)
from vadar.agents.modules.dist_helper import hl_gauss_target
from vadar.agents.modules.q_helper import (
    sample_mip_q_values,
    sample_mip_q_ensemble_values,
    compute_mip_q_predictions,
    compute_mip_q_ensemble_predictions,
)


class SACVadarAgent(flax.struct.PyTreeNode):
    """Soft Actor-Critic (SAC) agent with a per-subgoal (VaDaR) critic.

    The actor is shared between all Q functions of different chunk lengths.
    The actor always predicts action chunks, and is shared between all Q functions.
    There are chunked critics and non-chunked critics.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the SAC critic loss. Supports both standard ensemble Q and MIP-Q critics."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]  # take the first action

        # Sample next actions with current policy
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.network.select('actor')(batch['next_observations'])
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=sample_rng)

        G = self.config['num_subgoals']
        use_mip_q = self.config.get('use_mip_q', False)
        if use_mip_q:
            noise_scale = self.config.get('mip_q_noise_scale', 1.0)
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            target_critic_fn = self.network.select('target_critic')
            bsz = batch['next_observations'].shape[0]
            rng, noise_rng = jax.random.split(rng)
            # VaDaR: MIQ is the explicit ensemble ValueMIPEnsemble(num_subgoals=G) only.
            noise_sample = jax.random.uniform(
                noise_rng, (bsz, G), minval=-noise_scale, maxval=noise_scale,
            )
            next_qs = sample_mip_q_ensemble_values(
                critic_fn=target_critic_fn,
                observations=batch['next_observations'],
                actions=next_actions,
                noise_sample=noise_sample,
                mip_t_star=mip_t_star,
            )  # (G, M, B)
        else:
            next_qs = self.network.select('target_critic')(
                batch['next_observations'], actions=next_actions,
            )  # (G, M, B)
        # VaDaR: aggregate over the ensemble axis (axis=1) -> (G, B)
        if self.config['q_agg'] == 'min':
            next_q = jnp.min(next_qs, axis=1)
        else:
            next_q = jnp.mean(next_qs, axis=1)

        # Get temperature for entropy bonus
        temp = self.network.select('temp')()

        # SAC target with optional entropy bonus (applied to every subgoal's next value)
        if self.config.get('entropy_backup', False):
            next_q = next_q - temp * next_log_probs[None, :]

        # VaDaR: all-subgoal TD targets (G, B) from the subgoal dones at the end of the chunk
        if self.config.get('use_subgoal_discounts', False):
            discount = subgoal_discount_factors(self.config, G)
        else:
            discount = self.config['discount']
        target_q = compute_vadar_td_target(
            dones=batch['subgoal_dones'][:, -1, :],
            next_q=next_q,
            masks=batch['masks'][..., -1],
            discount=discount,
            horizon_length=self.config["horizon_length"],
            head_types=self.config.get('subgoal_head_types'),
        )

        if use_mip_q:
            critic_base_fn = self.network.select('critic')

            def critic_fn_with_params(obs, actions, scalar_input, time, is_encoded, return_logits):
                return critic_base_fn(
                    obs,
                    actions=actions,
                    scalar_input=scalar_input,
                    time=time,
                    params=grad_params,
                    is_encoded=is_encoded,
                    return_logits=return_logits,
                )

            rng, mip_rng = jax.random.split(rng)
            q_0, q = compute_mip_q_ensemble_predictions(
                critic_fn=critic_fn_with_params,
                observations=batch['observations'],
                actions=batch_actions,
                target_q=target_q,
                rng=mip_rng,
                noise_scale=noise_scale,
                mip_t_star=mip_t_star,
                return_logits=False,
            )  # (G, M, B), (G, M, B)
            td_err_0 = jnp.square(q_0 - target_q[:, None, :])
            td_err_t = jnp.square(q   - target_q[:, None, :])
            critic_loss = ((td_err_0 + td_err_t) * batch['valid'][..., -1]).mean()
        else:
            # Current Q-values (standard ensemble path): (G, M, B)
            q = self.network.select('critic')(
                batch['observations'], actions=batch_actions, params=grad_params,
            )
            td_err = jnp.square(q - target_q[:, None, :])
            critic_loss = (td_err * batch['valid'][..., -1]).mean()

        # VaDaR: per-subgoal TD errors (G,), valid-weighted means over (M, B) so
        # they decompose critic_loss.
        v = batch['valid'][..., -1].astype(jnp.float32)  # (B,)
        if use_mip_q:
            denom = v.sum() * td_err_0.shape[1] + 1e-8
            td_err_0_per_g = (td_err_0 * v).sum(axis=(1, 2)) / denom
            td_err_t_per_g = (td_err_t * v).sum(axis=(1, 2)) / denom
            td_err_per_g = td_err_0_per_g + td_err_t_per_g
        else:
            denom = v.sum() * td_err.shape[1] + 1e-8
            td_err_per_g = (td_err * v).sum(axis=(1, 2)) / denom

        info = {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'target_q_mean': target_q.mean(),
            'next_log_probs_mean': next_log_probs.mean(),
            'temp': temp,
            'td_err_total': td_err_per_g.sum(),
        }
        for g in range(G):
            info[f'td_err_subgoal_{g}'] = td_err_per_g[g]
        if use_mip_q:
            info['td_err0_total'] = td_err_0_per_g.sum()
            info['td_err_t_total'] = td_err_t_per_g.sum()
            for g in range(G):
                info[f'td_err0_subgoal_{g}'] = td_err_0_per_g[g]
                info[f'td_err_t_subgoal_{g}'] = td_err_t_per_g[g]
        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute the SAC actor loss with integrated temperature loss."""
        # Sample actions with current policy
        rng, sample_rng = jax.random.split(rng)
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=sample_rng)

        # Get Q-values for sampled actions (MIP-Q two-step or plain critic forward)
        if self.config.get('use_mip_q', False):
            G = self.config['num_subgoals']
            noise_scale = self.config.get('mip_q_noise_scale', 1.0)
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            critic_fn = self.network.select('critic')
            bsz = batch['observations'].shape[0]
            rng, noise_rng = jax.random.split(rng)
            noise_sample = jax.random.uniform(
                noise_rng, (bsz, G), minval=-noise_scale, maxval=noise_scale,
            )
            qs = sample_mip_q_ensemble_values(
                critic_fn=critic_fn,
                observations=batch['observations'],
                actions=actions,
                noise_sample=noise_sample,
                mip_t_star=mip_t_star,
            )  # (G, M, B)
        else:
            qs = self.network.select('critic')(batch['observations'], actions=actions)  # (G, M, B)

        if self.config['q_agg'] == 'min':
            q_per_subgoal = jnp.min(qs, axis=1)
        else:
            q_per_subgoal = jnp.mean(qs, axis=1)

        # VaDaR: Q_agg = sum_g w(d)[g] Q(s, a, g), weights from the chunk-end subgoal dones
        weights = subgoal_weights(
            batch['subgoal_dones'][:, -1, :],
            self.config['weighting'],
            self.config.get('subgoal_dag'),
            self.config.get('subgoal_weights'),
        )  # (G, B)
        q = aggregate_subgoal_values(q_per_subgoal, weights)  # (B,)

        # Get temperature
        temp = self.network.select('temp')()

        # SAC actor loss: maximize Q - temperature * entropy
        actor_loss = jnp.mean((temp * log_probs - q) * batch['valid'][..., -1])

        # Temperature loss
        temp_param = self.network.select('temp')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs * batch['valid'][..., -1]).mean()
        temp_loss = (temp_param * (entropy - self.config['target_entropy'])).mean()

        total_loss = actor_loss + temp_loss

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'temp_loss': temp_loss,
            'log_probs_mean': log_probs.mean(),
            'entropy': -log_probs.mean(),
            'q_actor_mean': q.mean(),
            'temp': temp,
            'subgoal_weights_mean': weights.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch, batch_success=None, success_flag=False):
        """Update the agent over a batch of transitions.

        Extra args (batch_success, success_flag) accepted for runner
        compatibility but are unused — SAC has no success-buffer logic.
        """
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    def bc_loss(self, batch, grad_params, rng):
        """Compute BC loss: negative log-prob of dataset actions under the actor."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        # Clip actions away from ±1 to avoid atanh(±1)=±inf in TanhNormal log_prob.
        batch_actions = jnp.clip(batch_actions, -1.0 + 1e-6, 1.0 - 1e-6)
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        log_probs = dist.log_prob(batch_actions)
        bc_loss = -(log_probs * batch['valid'][..., -1]).mean()
        return bc_loss, {'bc_loss': bc_loss, 'bc_log_probs': log_probs.mean()}

    @jax.jit
    def bc_update(self, batch):
        """Update actor via behavioral cloning."""
        new_rng, rng = jax.random.split(self.rng)
        def loss_fn(grad_params):
            return self.bc_loss(batch, grad_params, rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        # Unprefixed keys, like upstream OGPO: bc_runner logs this dict under
        # "offline_agent" and looks up 'bc_loss' for its loss trace.
        return self.replace(network=new_network, rng=new_rng), info

    @staticmethod
    def _q_warmup_update(agent, batch):
        """Train critic only via TD learning (no actor update)."""
        new_rng, rng = jax.random.split(agent.rng)
        def loss_fn(grad_params):
            return agent.critic_loss(batch, grad_params, rng)
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def batch_q_warmup_update(self, batch):
        """Batch Q warmup: critic-only updates over a scan."""
        agent, infos = jax.lax.scan(self._q_warmup_update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def batch_critic_only_update(self, batch):
        """Critic-only updates (used by online_rl_runner when utd_q > utd_pi)."""
        agent, infos = jax.lax.scan(self._q_warmup_update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def get_subgoal_q_values(self, observations, actions):
        """VaDaR: per-subgoal Q-values (G, B), aggregated over the ensemble.

        Args:
            observations: (B, obs_dim)
            actions: (B, H, action_dim) action chunks (or already flattened).
        """
        flat_actions = actions.reshape(observations.shape[0], -1)
        if self.config.get('use_mip_q', False):
            # Deterministic evaluation: zero noise.
            noise_sample = jnp.zeros((observations.shape[0], self.config['num_subgoals']))
            qs = sample_mip_q_ensemble_values(
                critic_fn=self.network.select('critic'),
                observations=observations,
                actions=flat_actions,
                noise_sample=noise_sample,
                mip_t_star=self.config.get('mip_q_t_star', 0.9),
            )  # (G, M, B)
        else:
            qs = self.network.select('critic')(observations, actions=flat_actions)  # (G, M, B)
        if self.config['q_agg'] == 'min':
            return jnp.min(qs, axis=1)
        return jnp.mean(qs, axis=1)

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
        **kwargs,
    ):
        """Sample stochastic actions from the actor policy."""
        dist = self.network.select('actor')(observations)
        actions = dist.sample(seed=rng)
        return actions

    @jax.jit
    def compute_flow_actions(self, observations, **kwargs):
        """Deterministic action selection (policy mean).

        SAC has no flow model; this returns the deterministic policy mean.
        Provided so the BC/online_rl_runner eval path works unchanged.
        """
        dist = self.network.select('actor')(observations)
        if hasattr(dist, 'mode'):
            return dist.mode()
        return dist.mean()

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new SAC agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        # Promote to ≥2D for network init. main.py passes single-example
        # tensors (shape `(D,)`) via Dataset.sample(()); plain Value tolerates
        # 1D inputs but ValueMIPEnsemble's tile path requires a batch dim.
        if ex_observations.ndim == 1:
            ex_observations = ex_observations[None]
        if ex_actions.ndim == 1:
            ex_actions = ex_actions[None]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # VaDaR: lists/dicts from YAML become tuples so the (static) config stays hashable.
        G = int(config['num_subgoals'])
        for key in ('subgoal_weights', 'subgoal_head_types', 'subgoal_discounts'):
            if config.get(key) is not None:
                config[key] = tuple(config[key])
        if config.get('subgoal_dag') is not None:
            config['subgoal_dag'] = {int(k): tuple(int(p) for p in v) for k, v in dict(config['subgoal_dag']).items()}
        if config.get('subgoal_head_types') is not None:
            assert len(config['subgoal_head_types']) == G, "subgoal_head_types needs one entry per subgoal"

        # Define networks. VaDaR: every critic returns (G, M, B).
        if config.get('use_mip_q', False):
            assert config.get('mip_q_ensemble', False), \
                "VaDaR MIQ needs mip_q_ensemble=true (ValueMIP has no per-subgoal heads)"
            from vadar.networks import ValueMIPEnsemble
            critic_def = ValueMIPEnsemble(
                hidden_dims=config['value_hidden_dims'],
                num_ensembles=config.get('num_ensemble_members', 10),
                mip_q_noise_scale=config.get('mip_q_noise_scale', 1.0),
                mip_t_star=config.get('mip_q_t_star', 0.9),
                layer_norm=config['layer_norm'],
                encoder=encoders.get('critic'),
                critic_loss_type=config['critic_loss_type'],
                num_bins=config.get('num_bins', 256),
                q_min=config.get('q_min', None),
                q_max=config.get('q_max', None),
                num_subgoals=G,
            )
            print(f"[MIP-Q] SAC-VaDaR using ValueMIPEnsemble (G={G}, M={config.get('num_ensemble_members', 10)}, "
                  f"noise={config.get('mip_q_noise_scale', 1.0)}, t*={config.get('mip_q_t_star', 0.9)})")
        else:
            assert encoders.get('critic') is None, "VaDaR critics do not take an observation encoder"
            assert config['critic_loss_type'] == 'mse', "VaDaR critics support the MSE loss only"
            critic_def = VADAR_CRITICS[config.get('vadar_critic', 'goal_conditioned')](
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=config['num_qs'],
                num_subgoals=G,
            )

        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
            tanh_squash=True,
            state_dependent_std=True,
        )

        # Wrap with GoalConditionedNetwork when observations carry a goal suffix
        # (`goal_obs`: HGC / iHGC / HER; see docs/DESIGN.md).
        if config.get('goal_obs'):
            # HER 'state'/'latent' goals set goal_dim (goal-space width);
            # subgoal-vector goals fall back to num_subgoals.
            goal_dim = int(config.get('goal_dim') or config['num_subgoals'])
            goal_embed_dim = config.get('goal_embed_dim', 32)
            actor_def = GoalConditionedNetwork(
                inner=actor_def,
                goal_dim=goal_dim,
                goal_embed_dim=goal_embed_dim,
                use_goals=config.get('goal_conditioned_actor', True),
            )
            critic_def = GoalConditionedNetwork(
                inner=critic_def,
                goal_dim=goal_dim,
                goal_embed_dim=goal_embed_dim,
                use_goals=config.get('goal_conditioned_critic', True),
            )
            print(f"[GC] Wrapped actor (use_goals={config.get('goal_conditioned_actor', True)}) "
                  f"and critic (use_goals={config.get('goal_conditioned_critic', True)}) "
                  f"with GoalConditionedNetwork (goal_dim={goal_dim}, embed_dim={goal_embed_dim}, "
                  f"goal_obs={config.get('goal_obs')})")

        temp_def = LogParam(init_value=config.get('init_temperature', 1.0))
        # Build init kwargs per network. MIP-Q critics need a `rng` kwarg
        # for noise sampling during init (otherwise raise "must provide
        # scalar_input or rng"). Standard Value critics ignore it.
        critic_init_kw = dict(observations=ex_observations, actions=full_actions)
        target_critic_init_kw = dict(critic_init_kw)
        if config.get('use_mip_q', False):
            init_rng, critic_init_rng, target_critic_init_rng = jax.random.split(init_rng, 3)
            critic_init_kw['rng'] = critic_init_rng
            target_critic_init_kw['rng'] = target_critic_init_rng

        network_info = dict(
            actor=(actor_def, (ex_observations,)),
            critic=(critic_def, critic_init_kw),
            target_critic=(copy.deepcopy(critic_def), target_critic_init_kw),
            temp=(temp_def, ()),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        # Set target entropy if not specified (using action_dim / 2)
        if config.get('target_entropy') is None:
            config['target_entropy'] = -float(full_action_dim) / 2

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
