from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from vadar.utils.augmentations import batched_random_crop, color_transform

eps = 1e-9  # Consistent epsilon used throughout

def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


class LatentGoalEncoder:
    """Numpy forward of the frozen SSL bottleneck encoder saved by
    scripts/rollout/train_subgoal_classifier.py. Maps raw base obs -> z."""

    def __init__(self, path=None, arrays=None):
        d = dict(np.load(path)) if path is not None else arrays
        self.arrays = {k: np.asarray(d[k]) for k in
                       ('W1', 'b1', 'W2', 'b2', 'Wz', 'bz', 'mu', 'sigma')}
        self.eps = float(d['eps'])
        self.embed_dim = int(d['embed_dim'])

    def __call__(self, x):
        a = self.arrays
        h = np.maximum((np.asarray(x, np.float32) - a['mu']) / a['sigma']
                       @ a['W1'] + a['b1'], 0)
        h = np.maximum(h @ a['W2'] + a['b2'], 0)
        return h @ a['Wz'] + a['bz']

    def success(self, achieved_z, desired_z):
        """1.0 where ||z - z_g|| <= eps (broadcast over leading dims)."""
        d = np.linalg.norm(np.asarray(achieved_z) - np.asarray(desired_z), axis=-1)
        return (d <= self.eps).astype(np.float32)


def her_reward(achieved, desired, pos_blocks, bool_dims, pos_threshold):
    """HER goal-space success predicate, vectorized over leading batch dims.

    Lives here (not envs/) so HerReplayBuffer can use it without a circular
    import. Success requires every 3-D position block within `pos_threshold`
    (Euclidean) AND every boolean dim to match at the 0.5 level. Returns
    float {0., 1.}; the Fetch-convention reward is `her_reward(...) - 1.0`.
    """
    achieved = np.asarray(achieved, dtype=np.float32)
    desired = np.asarray(desired, dtype=np.float32)
    ok = np.ones(np.broadcast_shapes(achieved.shape, desired.shape)[:-1], dtype=bool)
    for (s, e) in pos_blocks:
        ok &= np.linalg.norm(achieved[..., s:e] - desired[..., s:e], axis=-1) <= pos_threshold
    for b in bool_dims:
        ok &= (achieved[..., b] > 0.5) == (desired[..., b] > 0.5)
    return ok.astype(np.float32)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, numpy_rng=None, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            numpy_rng: NumPy random generator for reproducible sampling. If None, creates default.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data, numpy_rng=numpy_rng)

    def __init__(self, *args, numpy_rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.

        # Initialize NumPy random generator for reproducible sampling
        if numpy_rng is None:
            self.numpy_rng = np.random.default_rng(42)  # Default seed for backward compatibility
        else:
            self.numpy_rng = numpy_rng

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        # JAX RNG for color augmentation — derive from the numpy_rng for reproducibility
        self.rng = jax.random.PRNGKey(int(self.numpy_rng.integers(2**31)))

        # ---- NEW: per-trajectory success -> scores (1 for successful traj, eps otherwise) ----
        if 'is_success' in self._dict:
            traj_success_mask = np.zeros(self.size, dtype=np.bool_)
            for i in range(len(self.terminal_locs)):
                s = self.initial_locs[i]
                e = self.terminal_locs[i]
                succ = np.any(self['is_success'][s:e+1] > 0)
                traj_success_mask[s:e+1] = succ
            new_scores = np.where(traj_success_mask, 1.0, eps).astype(np.float32)
            self._dict['score'] = new_scores  # ensure score reflects trajectory success
            self._traj_success_mask = traj_success_mask  # cached for filtering
        else:
            # Fallback: if no is_success, keep existing scores if any; cache empty mask.
            self._traj_success_mask = np.zeros(self.size, dtype=np.bool_)

    def compute_mc_returns(self, discount):
        mc_returns = np.zeros_like(self['rewards'])
        for i in range(len(self.terminal_locs)):
            start_idx = self.initial_locs[i]
            end_idx = self.terminal_locs[i]
            
            # Compute returns backward from end of trajectory
            G = 0.0
            for t in range(end_idx, start_idx - 1, -1):
                G = self['rewards'][t] + discount * G
                mc_returns[t] = G
        return mc_returns

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices (proportional to score if available)."""
        return self.numpy_rng.integers(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None, success_only=False):
        """Sample a batch of transitions.

        NOTE:
            - By default this is **uniform** over the buffer.
            - Score-based sampling is handled explicitly in `sample_sequence`
              via the `by_score` flag. We intentionally keep `sample` simple
              and unbiased so that online RL batches are not silently
              success-weighted.
        
        Args:
            batch_size: Number of samples to draw
            idxs: Optional pre-specified indices
            success_only: If True, sample only from successful transitions
        """
        if idxs is None:
            if success_only:
                # Filter to successful transitions
                success_mask = self._traj_success_mask[:self.size]
                success_idxs = np.where(success_mask)[0]
                if len(success_idxs) < batch_size:
                    raise ValueError(f"Not enough successful transitions: {len(success_idxs)} < {batch_size}")
                # Sample from successful indices
                idxs = self.numpy_rng.choice(success_idxs, size=batch_size, replace=False)
            else:
                # Uniform sampling over [0, self.size)
                idxs = self.numpy_rng.integers(self.size, size=batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None and batch['observations'].ndim > 3:
            # Apply random-crop image augmentation (legacy path for image-as-observation).
            if self.numpy_rng.random() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        # Augment separate image fields when present (Approach B)
        if self.p_aug is not None and 'images' in batch and batch['images'].ndim > 3:
            if self.numpy_rng.random() < self.p_aug:
                self.augment(batch, ['images', 'next_images'])
        # Include MC returns when available (used by q_warmup with mc_regression)
        if 'mc_returns' in self._dict and 'mc_returns' not in batch:
            batch['mc_returns'] = self['mc_returns'][idxs]
        return batch

    def sample_sequence(self, batch_size, sequence_length, discount, ret_next_act=False, ret_mc=False, filter_fn=None, success_only=None, by_score=False):
        """
        Sample sequences. Supports:
        - success_only: if True, only from successful trajectories; if False, only unsuccessful; if None, no success filter.
        - filter_fn: additional filtering function (e.g., score-based) over valid start positions.
            Signature: filter_fn(data_dict, num_starts) -> boolean mask of shape [num_starts]
        - by_score: if True, sample start indices proportionally to their score values.
        """
        B = batch_size
        T = sequence_length
        max_start = self.size - T  # FIXED: was self.size - T - 1
        if max_start < 0:
            raise ValueError(f"Sequence length too long for dataset size. Need at least {T} transitions, but have {self.size}")

        # Candidate start indices [0..max_start]
        starts = np.arange(max_start + 1, dtype=np.int64)

        # Base mask: all starts valid
        mask = np.ones(max_start + 1, dtype=bool)

        # Optional success-based filtering (per-trajectory)
        if success_only is not None:
            succ_mask = self._traj_success_mask[:max_start + 1]
            mask &= (succ_mask if success_only else ~succ_mask)

        # Optional extra filter (e.g., score-based thresholding)
        if filter_fn is not None:
            extra = filter_fn(self._dict, max_start + 1)
            if extra is None or len(extra) != max_start + 1:
                raise ValueError("filter_fn must return a boolean mask of length max_start + 1.")
            mask &= extra

        valid_indices = starts[mask]
        if len(valid_indices) == 0:
            raise ValueError("No transitions found matching filter")

        # Importance sampling over start indices proportional to score at the start
        # ONLY if by_score=True
        probs = None  # Default to uniform sampling
        if by_score:
            scores = self._dict.get('score', None)
            if scores is not None:
                w = np.asarray(scores[valid_indices], dtype=np.float64)
                total = w.sum()
                if np.isfinite(total) and total > 0:
                    probs = w / total
                # else: probs stays None (uniform)

        idxs = self.numpy_rng.choice(valid_indices, size=B, p=probs)

        # 2) Build a (B, T) index matrix and gather everything at once
        offs = np.arange(T, dtype=np.int64)[None, :]          # (1, T)
        seq_idxs = idxs[:, None].astype(np.int64) + offs      # (B, T)

        obs_seq         = self['observations'][seq_idxs]      # (B, T, obs_dim)
        next_obs_seq    = self['next_observations'][seq_idxs] # (B, T, obs_dim)
        actions_seq     = self['actions'][seq_idxs]           # (B, T, act_dim)
        rewards_seq     = self['rewards'][seq_idxs]           # (B, T)
        masks_seq       = self['masks'][seq_idxs]             # (B, T)
        terminals_seq   = self['terminals'][seq_idxs]         # (B, T)

        # 3) Running mask/terminal over time
        #    (note: np.minimum/maximum are ufuncs; .accumulate supports axis)
        masks_prefix     = np.minimum.accumulate(masks_seq, axis=1)
        terminals_prefix = np.maximum.accumulate(terminals_seq, axis=1)

        # 4) Valid: 1 at i==0; for i>0 it's 1 - terminals_prefix at i-1
        valid = np.ones_like(masks_seq, dtype=np.float32)
        valid[:, 1:] = 1.0 - terminals_prefix[:, :-1]

        # 5) Prefix discounted return (same semantics as your loop)
        #    No stop-at-terminal masking (matches original).
        rdtype = rewards_seq.dtype
        disc_pows = (discount ** np.arange(T)).astype(rdtype, copy=False)  # (T,)
        rewards_prefix = np.cumsum(rewards_seq * disc_pows[None, :], axis=1)

        # 6) "observations" = first frame only (keep your original API)
        first_obs = self['observations'][idxs]  # (B, obs_dim)
        last_obs = next_obs_seq[:, -1, ...]

        if self.p_aug is not None and first_obs.ndim > 3 and self.numpy_rng.random() < self.p_aug:
            tmp = {'observations': first_obs, 'next_observations': last_obs}
            self.augment(tmp, ['observations', 'next_observations'])
            first_obs = tmp['observations']
            last_obs = tmp['next_observations']

        result = dict(
            observations=first_obs.copy(),          # matches your return
            actions=actions_seq,
            masks=masks_prefix,
            rewards=rewards_prefix,
            terminals=terminals_prefix,
            valid=valid,
            next_observations=last_obs,
        )

        # Gather separate image fields when present (Approach B)
        if 'images' in self._dict:
            result['images'] = self['images'][idxs]                 # (B, H, W, C)
            result['next_images'] = self['next_images'][seq_idxs[:, -1]]  # (B, H, W, C)
            if self.p_aug is not None and self.numpy_rng.random() < self.p_aug:
                self.augment(result, ['images', 'next_images'])

        if 'full_states' in self._dict:
            result['full_states'] = self['full_states'][idxs]
            result['next_full_states'] = self['next_full_states'][seq_idxs[:, -1]]

        if 'is_success' in self._dict:
            result['is_success'] = self['is_success'][seq_idxs]

        # VaDaR: per-step subgoal completion indicators, (B, T, G).
        if 'subgoal_dones' in self._dict:
            result['subgoal_dones'] = self['subgoal_dones'][seq_idxs]
    
        if ret_next_act:
            result['next_actions'] = self['actions'][np.minimum(seq_idxs + 1, self.size - 1)]

        if 'mc_returns' in self._dict:
            result['mc_returns'] = self['mc_returns'][idxs]
        return result

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = self.numpy_rng.integers(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: batched_random_crop(arr, crop_froms, padding) if len(arr.shape) == 4 else arr,
                batch[key],
            )
            self.rng, aug_rng = jax.random.split(self.rng)
            batch[key] = (color_transform(aug_rng, batch[key] / 255.0) * 255.0).astype(jnp.uint8)


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size, numpy_rng=None):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
            numpy_rng: NumPy random generator for reproducible sampling. If None, creates default.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        if 'is_success' not in buffer_dict:
            buffer_dict['is_success'] = np.zeros(size, dtype=np.float32)
        if 'score' not in buffer_dict:
            buffer_dict['score'] = np.zeros(size, dtype=np.float32)
        return cls(buffer_dict, numpy_rng=numpy_rng)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size, numpy_rng=None):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
            numpy_rng: NumPy random generator for reproducible sampling. If None, creates default.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        init_size = get_size(init_dataset)
        
        if 'is_success' not in buffer_dict:
            buffer_dict['is_success'] = np.zeros(size, dtype=np.float32)
            if 'is_success' in init_dataset:
                buffer_dict['is_success'][:init_size] = init_dataset['is_success']
        if 'score' not in buffer_dict:
            buffer_dict['score'] = np.zeros(size, dtype=np.float32)
            if 'score' in init_dataset:
                buffer_dict['score'][:init_size] = init_dataset['score']
            elif 'is_success' in init_dataset:
                buffer_dict['score'][:init_size] = init_dataset['is_success']
        
        dataset = cls(buffer_dict, numpy_rng=numpy_rng)
        dataset.size = dataset.pointer = init_size
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

        self._traj_success_mask = np.zeros(self.max_size, dtype=np.bool_)

    def __reduce__(self):
        """Override reduce to force use of our getstate/setstate without calling __init__."""
        return (self.__class__.__new__, (self.__class__,), self.__getstate__())

    def __getstate__(self):
        """Support pickling."""
        state = self.__dict__.copy()
        # Ensure the internal dictionary data is saved. 
        # FrozenDict content might not be in __dict__.
        if '_dict' not in state and hasattr(self, '_dict'):
             state['_dict'] = self._dict
        # Save numpy_rng state for pickling
        if 'numpy_rng' in state:
            state['numpy_rng_state'] = state['numpy_rng'].__getstate__()
            del state['numpy_rng']  # Remove unpicklable object
        return state

    def __setstate__(self, state):
        """Support unpickling."""
        # Restore numpy_rng from saved state
        if 'numpy_rng_state' in state:
            numpy_rng_state = state.pop('numpy_rng_state')
            state['numpy_rng'] = np.random.default_rng()
            state['numpy_rng'].__setstate__(numpy_rng_state)
        
        # Restore all attributes including _dict
        for key, value in state.items():
            object.__setattr__(self, key, value)
        
        # Reset hash to ensure validity if needed by FrozenDict
        object.__setattr__(self, '_hash', None)

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""
        transition = transition.copy()
        if 'is_success' in transition and 'score' not in transition:
            # USE eps for unsuccessful steps instead of 0.0
            transition['score'] = 1.0 if float(transition['is_success']) > 0 else eps
        elif 'is_success' not in transition:
            transition['is_success'] = 0.0
            transition['score'] = eps
        
        # Optional: Uncomment for debugging
        # if 'is_success' in transition and 'score' in transition:
        #     is_success = float(transition['is_success'])
        #     score = float(transition['score'])
        #     if is_success == 1:
        #         assert score >= 1.0, f"When is_success == 1, score should be >= 1.0, got {score}"
        #     else:
        #         assert abs(score - eps) < 1e-6, f"When is_success != 1, score should be equal to epsilon ({eps}), got {score}"

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self._traj_success_mask[self.pointer] = (float(transition['is_success']) > 0)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


class HerReplayBuffer(ReplayBuffer):
    """Hindsight Experience Replay buffer mirroring SB3's HerReplayBuffer.

    SB3 semantics preserved:
      - only real transitions are stored; relabeled ("virtual") samples are
        created at sample time,
      - her_ratio = 1 - 1/(n_sampled_goal + 1) of each sampled batch is
        virtual (default 0.8), split exactly like SB3 (first slice virtual),
      - goal_selection_strategy in {'future' (inclusive), 'final', 'episode'},
      - only the desired-goal obs suffix and the rewards change; masks /
        terminals / valid are untouched (SB3 leaves dones unchanged),
      - ep_start / ep_length arrays track episode spans; episodes partially
        overwritten by ring wraparound are invalidated (ep_length = 0).

    Adaptation for this repo's chunked critics: the sampled unit is a T-step
    window, so one relabeled goal applies to the whole chunk (goal sampled
    from the chunk-start's episode, 'future' relative to the chunk start) and
    per-step rewards are recomputed before the discounted prefix sum.

    The desired goal is assumed to be the LAST `goal_dim` dims of the
    observation vector (the GoalConditionedNetwork suffix); the achieved goal
    is `base_obs[her_goal_indices]` (or the latent encoder's z).

    her_config keys: 'latent' (bool) + 'encoder_arrays' for latent goals, else
    'goal_indices', 'pos_blocks', 'bool_dims', 'pos_threshold' (0.05),
    'goal_dim' (len(goal_indices)); plus 'n_sampled_goal' (4) and
    'goal_selection_strategy' ('future').
    """

    @classmethod
    def create(cls, transition, size, numpy_rng=None, her_config=None):
        assert her_config is not None, "HerReplayBuffer.create requires her_config"
        buf = super().create(transition, size, numpy_rng=numpy_rng)
        buf._init_her(her_config)
        return buf

    def _init_her(self, her_config):
        self.her_latent = bool(her_config.get('latent', False))
        if self.her_latent:
            # Latent goals: z = encoder(base obs); success = ||z-z_g|| <= eps.
            self.her_encoder = LatentGoalEncoder(arrays=her_config['encoder_arrays'])
            self.her_goal_dim = self.her_encoder.embed_dim
            self.her_goal_indices = None
        else:
            self.her_goal_indices = np.asarray(her_config['goal_indices'], dtype=np.int64)
            self.her_pos_blocks = tuple(tuple(b) for b in her_config['pos_blocks'])
            self.her_bool_dims = tuple(her_config['bool_dims'])
            self.her_pos_threshold = float(her_config.get('pos_threshold', 0.05))
            self.her_goal_dim = int(her_config.get('goal_dim', len(self.her_goal_indices)))
        self.n_sampled_goal = int(her_config.get('n_sampled_goal', 4))
        self.her_ratio = 1.0 - 1.0 / (self.n_sampled_goal + 1)
        self.goal_selection_strategy = str(her_config.get('goal_selection_strategy', 'future'))
        assert self.goal_selection_strategy in ('future', 'final', 'episode')
        self.ep_start = np.zeros(self.max_size, dtype=np.int64)
        self.ep_length = np.zeros(self.max_size, dtype=np.int64)
        self._current_ep_start = 0

    def add_transition(self, transition):
        # SB3 wraparound semantics: an old episode partially overwritten by
        # the write pointer must never be sampled again.
        old_len = self.ep_length[self.pointer]
        if old_len > 0:
            old_start = self.ep_start[self.pointer]
            old_idxs = np.arange(old_start, old_start + old_len) % self.max_size
            self.ep_length[old_idxs] = 0

        write_idx = self.pointer
        super().add_transition(transition)
        self.ep_length[write_idx] = 0  # invalid until its episode completes

        if float(transition['terminals']) > 0:
            start = self._current_ep_start
            length = (write_idx - start) % self.max_size + 1
            span = (start + np.arange(length)) % self.max_size
            self.ep_start[span] = start
            self.ep_length[span] = length
            self._current_ep_start = (write_idx + 1) % self.max_size

    def _sample_her_goals(self, v_idxs):
        """Sample one hindsight goal per virtual chunk start (SB3 formulas)."""
        ep_starts = self.ep_start[v_idxs]
        ep_lengths = self.ep_length[v_idxs]
        pos_in_ep = (v_idxs - ep_starts) % self.max_size
        if self.goal_selection_strategy == 'future':
            # Inclusive of the current transition, exactly like SB3.
            gpos = self.numpy_rng.integers(pos_in_ep, ep_lengths)
        elif self.goal_selection_strategy == 'final':
            gpos = ep_lengths - 1
        else:  # 'episode'
            gpos = self.numpy_rng.integers(0, ep_lengths)
        src = (ep_starts + gpos) % self.max_size
        G = self.her_goal_dim
        base = self['next_observations'][src][:, :-G]
        if self.her_latent:
            return self.her_encoder(base)
        return base[:, self.her_goal_indices]

    def sample_sequence(self, batch_size, sequence_length, discount, ret_next_act=False, ret_mc=False, filter_fn=None, success_only=None, by_score=False):
        """Dataset.sample_sequence with SB3-style HER relabeling.

        Returns the same fields as Dataset.sample_sequence; HER-specific
        insertions are marked with `# HER:` comments.
        """
        B = batch_size
        T = sequence_length
        max_start = self.size - T
        if max_start < 0:
            raise ValueError(f"Sequence length too long for dataset size. Need at least {T} transitions, but have {self.size}")

        starts = np.arange(max_start + 1, dtype=np.int64)
        mask = np.ones(max_start + 1, dtype=bool)

        # HER: only chunks starting inside a complete, non-invalidated episode
        # (mirrors SB3's `ep_length > 0` validity check).
        mask &= self.ep_length[:max_start + 1] > 0

        if success_only is not None:
            succ_mask = self._traj_success_mask[:max_start + 1]
            mask &= (succ_mask if success_only else ~succ_mask)

        if filter_fn is not None:
            extra = filter_fn(self._dict, max_start + 1)
            if extra is None or len(extra) != max_start + 1:
                raise ValueError("filter_fn must return a boolean mask of length max_start + 1.")
            mask &= extra

        valid_indices = starts[mask]
        if len(valid_indices) == 0:
            raise ValueError("No transitions found matching filter")

        probs = None
        if by_score:
            scores = self._dict.get('score', None)
            if scores is not None:
                w = np.asarray(scores[valid_indices], dtype=np.float64)
                total = w.sum()
                if np.isfinite(total) and total > 0:
                    probs = w / total

        idxs = self.numpy_rng.choice(valid_indices, size=B, p=probs)

        offs = np.arange(T, dtype=np.int64)[None, :]
        seq_idxs = idxs[:, None].astype(np.int64) + offs

        obs_seq         = self['observations'][seq_idxs]
        next_obs_seq    = self['next_observations'][seq_idxs]
        actions_seq     = self['actions'][seq_idxs]
        rewards_seq     = self['rewards'][seq_idxs]
        masks_seq       = self['masks'][seq_idxs]
        terminals_seq   = self['terminals'][seq_idxs]

        # HER: relabel the first `her_ratio` slice of the batch (SB3's split).
        # Fancy indexing above produced fresh arrays, so in-place writes are
        # safe. Only the desired-goal suffix and rewards change.
        # Success-buffer batches (success_only=True) are NOT relabeled —
        # hindsight goals would destroy the "successful under the real goal"
        # semantics those batches exist to provide.
        G = self.her_goal_dim
        nb_virtual = 0 if success_only else int(self.her_ratio * B)
        if nb_virtual > 0:
            goals = self._sample_her_goals(idxs[:nb_virtual])          # (nv, G)
            obs_seq[:nb_virtual, :, -G:] = goals[:, None, :]
            next_obs_seq[:nb_virtual, :, -G:] = goals[:, None, :]
            base = next_obs_seq[:nb_virtual, :, :-G]                   # (nv, T, base)
            if self.her_latent:
                achieved_z = self.her_encoder(base.reshape(-1, base.shape[-1])) \
                    .reshape(base.shape[0], base.shape[1], -1)
                succ = self.her_encoder.success(achieved_z, goals[:, None, :])
            else:
                achieved = base[:, :, self.her_goal_indices]           # (nv, T, G)
                succ = her_reward(achieved, goals[:, None, :],
                                  self.her_pos_blocks, self.her_bool_dims,
                                  self.her_pos_threshold)
            rewards_seq[:nb_virtual] = (succ - 1.0).astype(rewards_seq.dtype, copy=False)

        masks_prefix     = np.minimum.accumulate(masks_seq, axis=1)
        terminals_prefix = np.maximum.accumulate(terminals_seq, axis=1)

        valid = np.ones_like(masks_seq, dtype=np.float32)
        valid[:, 1:] = 1.0 - terminals_prefix[:, :-1]

        # HER: prefix discounted return masked to stop at the first terminal in
        # the chunk (as in the fork). HER rewards are dense (-1 per step), so a
        # chunk straddling an episode boundary must not add the next episode's
        # rewards. (Dataset.sample_sequence does not mask; with sparse rewards
        # the two agree.)
        rdtype = rewards_seq.dtype
        disc_pows = (discount ** np.arange(T)).astype(rdtype, copy=False)
        post_term = np.ones_like(rewards_seq, dtype=rdtype)
        post_term[:, 1:] = (1.0 - terminals_prefix[:, :-1]).astype(rdtype, copy=False)
        masked_rewards = rewards_seq * post_term
        rewards_prefix = np.cumsum(masked_rewards * disc_pows[None, :], axis=1)

        # HER: first_obs must come from the (relabeled) sequence, not a fresh
        # gather from the buffer.
        first_obs = obs_seq[:, 0, ...]
        last_obs = next_obs_seq[:, -1, ...]

        if self.p_aug is not None and first_obs.ndim > 3 and self.numpy_rng.random() < self.p_aug:
            tmp = {'observations': first_obs, 'next_observations': last_obs}
            self.augment(tmp, ['observations', 'next_observations'])
            first_obs = tmp['observations']
            last_obs = tmp['next_observations']

        result = dict(
            observations=first_obs.copy(),
            actions=actions_seq,
            masks=masks_prefix,
            rewards=rewards_prefix,
            terminals=terminals_prefix,
            valid=valid,
            next_observations=last_obs,
        )

        if 'images' in self._dict:
            result['images'] = self['images'][idxs]
            result['next_images'] = self['next_images'][seq_idxs[:, -1]]
            if self.p_aug is not None and self.numpy_rng.random() < self.p_aug:
                self.augment(result, ['images', 'next_images'])

        if 'full_states' in self._dict:
            result['full_states'] = self['full_states'][idxs]
            result['next_full_states'] = self['next_full_states'][seq_idxs[:, -1]]

        if 'is_success' in self._dict:
            result['is_success'] = self['is_success'][seq_idxs]

        # VaDaR: per-step subgoal completion indicators, (B, T, G).
        if 'subgoal_dones' in self._dict:
            result['subgoal_dones'] = self['subgoal_dones'][seq_idxs]

        if ret_next_act:
            result['next_actions'] = self['actions'][np.minimum(seq_idxs + 1, self.size - 1)]

        if 'mc_returns' in self._dict:
            result['mc_returns'] = self['mc_returns'][idxs]
        return result


def make_replay_buffer(rb_template, size, numpy_rng, her_config=None):
    """Online replay buffer: HerReplayBuffer when `her_config` is given, else ReplayBuffer."""
    if her_config is not None:
        buf = HerReplayBuffer.create(rb_template, size, numpy_rng=numpy_rng, her_config=her_config)
        print(f"[HER] HerReplayBuffer active: n_sampled_goal={buf.n_sampled_goal} "
              f"(her_ratio={buf.her_ratio:.2f}), strategy={buf.goal_selection_strategy}, "
              f"goal_dim={buf.her_goal_dim}")
        return buf
    return ReplayBuffer.create(rb_template, size, numpy_rng=numpy_rng)
