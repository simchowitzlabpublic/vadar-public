"""End-to-end CPU test of the online runner on a toy env that reports subgoal dones.

The toy env has G=3 sequential subgoals (|x| crossing 0.5, 1.0, 1.5 in the
first state coordinate); it checks that transitions carry `subgoal_dones`
into the replay buffer and that every agent family runs its online loop.
"""

import gymnasium
import numpy as np
import pytest
from gymnasium.spaces import Box

from vadar.utils.config_loader import load_config, config_to_agent_dict
from vadar.utils.datasets import Dataset
from vadar.utils.main_helper import LoggingHelper

OBS_DIM, ACT_DIM, H, G = 4, 2, 2, 3


class ToySubgoalEnv(gymnasium.Env):
    def __init__(self):
        self.observation_space = Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = Box(-1, 1, (ACT_DIM,), np.float32)

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros(OBS_DIM, np.float32)
        self.dones = np.zeros(G, np.float32)
        self.t = 0
        return self.x.copy(), {}

    def step(self, action):
        self.t += 1
        self.x[:ACT_DIM] += 0.3 * np.asarray(action, np.float32)
        self.dones = np.maximum(self.dones, (abs(self.x[0]) > np.array([0.5, 1.0, 1.5])).astype(np.float32))
        success = bool(self.dones[-1])
        info = {'success': float(success), 'subgoal_dones': self.dones.copy()}
        return self.x.copy(), float(success) - 1.0, success, self.t >= 30, info


def _dataset(n=400):
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(n, OBS_DIM)).astype(np.float32)
    terminals = np.zeros(n, np.float32)
    terminals[29::30] = 1.0
    return Dataset.create(
        observations=obs, next_observations=obs, actions=rng.uniform(-1, 1, (n, ACT_DIM)).astype(np.float32),
        rewards=-np.ones(n, np.float32), terminals=terminals, masks=np.ones(n, np.float32),
        subgoal_dones=(rng.uniform(size=(n, G)) < 0.3).astype(np.float32),
    )


@pytest.mark.parametrize("algo,warmup_mode", [
    ('ogpo_vadar', 'phase_split'), ('ogpo_vadar', 'interleaved'), ('fql_vadar', 'interleaved'),
    ('sac_vadar', 'interleaved'), ('ogpo', 'phase_split'),
])
def test_online_loop(algo, warmup_mode, tmp_path):
    from vadar.main import AGENTS
    from vadar.runners.online_rl_runner import run_online_rl
    config = load_config(algo, {
        'agent.actor_hidden_dims': [16], 'agent.value_hidden_dims': [16], 'agent.one_step_hidden_dims': [16],
        'agent.num_qs': 2, 'agent.batch_size': 4, 'agent.ppo_batch_size': 4, 'agent.group_num_samples': 2,
        'agent.best_of_n': 2, 'agent.horizon_length': H, 'agent.flow_steps': 2,
    })
    cfg = config_to_agent_dict(config)
    cfg.update(q_min=-30.0, q_max=15.0)
    if algo.endswith('_vadar'):
        cfg.update(num_subgoals=G, subgoal_discounts=None, use_subgoal_discounts=False)
    ds = _dataset()
    ex = ds.sample(())
    cls = AGENTS[algo]
    if algo.startswith('sac'):
        agent = cls.create(0, ex['observations'], ex['actions'], cfg)
    else:
        agent = cls.create(0, ex['observations'], ex['actions'], cfg, None)
    logger = LoggingHelper({}, None, log_enabled=False)
    agent, step, info, rb = run_online_rl(
        agent=agent, env=ToySubgoalEnv(), eval_env=ToySubgoalEnv(), eval_envs=None, train_dataset=ds,
        config=cfg, logger=logger, save_dir=str(tmp_path), algo_name=algo, online_steps=200,
        start_training=20, batch_size=4, horizon_length=H, discount=0.99, q_warmup_steps=60,
        q_warmup_mode=warmup_mode, warmup_offline_ratio=0.5 if warmup_mode == 'interleaved' else 0.0,
        use_success_buffer=True, success_buffer_batch_size=4, best_of_n=2, eval_interval=60,
        n_eval_envs=1, eval_episodes=1, log_interval=50, save_interval=-1, log_enabled=False, seed=0,
        buffer_size=1000,
    )
    assert 'subgoal_dones' in rb._dict and rb.size > 0
    assert rb['subgoal_dones'][:rb.size].max() >= 0.0
