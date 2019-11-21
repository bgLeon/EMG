# This code is a modified version of "https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py"

import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, s1, a, s2, next_goal):
        data = (s1, a, s2, next_goal)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        S1, A, S2, Goal = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            s1, a, s2, next_goal = data
            S1.append(np.array(s1, copy=False))
            A.append(np.array(a, copy=False))
            S2.append(np.array(s2, copy=False))
            Goal.append(np.array(next_goal, copy=False))
        return np.array(S1), np.array(A), np.array(S2), np.array(Goal)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

