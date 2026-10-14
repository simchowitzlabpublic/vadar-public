# MIQ + VaDaR-NF: the two-step minimal iterative Q critic (explicit ensemble of 10
# networks, uniform noise in [-1, 1], t* = 0.9); use_mip_q replaces the vadar_critic.
# Fork: mipQ-experiments/* (use_mip_q=true, mip_q_ensemble=true). The critic-UTD
# sweep adds --buffer.utd_q=<1|4> --buffer.utd_pi=1 (see sweep_miq_utd.sh).
ALGO=$VADAR_ALGO
METHOD_FLAGS=(
    --env.goal_obs=null
    --agent.weighting=next_future
    --agent.use_mip_q=true
    --agent.mip_q_ensemble=true
    --agent.mip_q_noise_scale=1.0
    --agent.mip_q_t_star=0.9
    --agent.num_ensemble_members=10
)
