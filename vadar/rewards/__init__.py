"""Preference-based reward baselines compared against VaDaR.

Dense (paper App. "Dense Preference Reward"; reference repo OGPO-pbrl-baseline)
    A state-only reward ensemble r(s) (3 MLPs, 4x512, leaky-relu, tanh output)
    is fit with the Bradley-Terry loss on whole-episode preferences: a
    successful episode beats a failed one, same-success pairs are dropped.
    Episodes are the offline demos plus ~300 policy rollouts from early / mid /
    late checkpoints (`rollout_dataset_path`, collected with
    scripts/pbrl/collect_preference_rollouts.py). Online, r(s_t) replaces the
    env reward of every transition of the sparse backbone (OGPO, FQL, SAC).

Pref-Subgoal (paper Sec. "TD critic learning outperforms preference-based
per-subgoal reward"; reference repo OGPO-pbrl-subgoal)
    One ensemble r(s, g) on [s, one_hot(g)]. For each subgoal g, an episode
    that completes g beats one that does not, and between two that do the
    earlier completion wins (`subgoal_label_mode: success_time`). Pairs are
    tagged with g and the BT loss conditions on it. Per step,
        r_t = sum_g w_NF(d_{t+1})[g] r(s_t, e_g)
    with VaDaR's next-future weighting on the post-step subgoal_dones. Demos
    only. The backbone's scalar critic trains on r_t.

    The first Pref-Subgoal runs added `subgoal_done_bonus * #done` (2.0) to
    every step. With termination on success, a positive per-step stream made
    stalling before the last subgoal optimal (an annuity worth more than
    finishing), and the agents completed S-1 subgoals and waited out the clock.
    The rerun with bonus 0.0 fixed this; 0.0 is the default here.

Both baselines (PEBBLE-style online updates)
    Every 5000 env steps: 128 new pairs from the finished online episodes
    (labelled by env success / subgoal_dones), FIFO online buffer (36k),
    1 epoch of fine-tuning on offline + online pairs, then every replay-buffer
    reward is relabelled with the new model.

Modules
    config.py                     resolve the `reward_model` config (configs/pbrl/*.yaml)
    preference_data.py            episode stores, labels, pair generation, rollout loading (numpy)
    preference_reward.py          reward network, BT training, pretraining, checkpoints (JAX)
    subgoal_preference_reward.py  VaDaR weightings in numpy + per-subgoal aggregation
    pebble.py                     learned reward, online updates, replay relabelling
    pbrl_setup.py                 prepare_pbrl(): one call from main.py
The online loop is vadar/runners/online_pbrl_runner.py.
"""
