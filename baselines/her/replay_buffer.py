import threading

import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions
        # self.sg_batch_sizes = np.zeros(3)

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size, policy_index):# ts, policy_index):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        ts = np.where(buffers['info_goal_index'] == policy_index)
        ts = [ts[1][np.nonzero(ts[0] == j)[0]] for j in range(buffers['u'].shape[0])]
        ep_Ts = np.array([len(t) for t in ts])
        cand_eps = np.where(ep_Ts > 1)[0]
        #If no cand_eps then no need to sample
        if len(cand_eps) == 0:
            # print("no cand eps")
            return None
        num_sg_ts = np.sum(ep_Ts)
        sg_batch_size = (num_sg_ts / (self.current_size * (self.T-1)) * batch_size).astype(int)

        # print("REPLAY SAMPLE CALLED")
        transitions = self.sample_transitions(buffers, sg_batch_size, ts, cand_eps)#, [buffers['u'].shape[1]]*buffers['u'].shape[0])
        # print("REPLAY SAMPLE OUT")

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        #Make it so we sample from each subgoal proportionate to the amount of timesteps spent on each subgoal in last episode batch
        # Account for 1 len episodes? - or caught/catch above anyway
        # self.sg_batch_sizes = [sum(len(np.where(episode_batch['info_goal_index'] == i))) for i in range(3)]
        

        #change to num_policies
        # for i in range(len(self.sg_batch_sizes)):
        #     ts = np.where(episode_batch['info_goal_index'] == i)
        #     # ts = [ts[1][np.nonzero(ts[0] == j)[0]] for j in range(buffers['u'].shape[0])]
        #     ep_Ts = np.array([len(t) for t in ts])
        #     cand_eps = np.where(ep_Ts > 1)[0]
        #     #If no cand_eps then no need to sample
        #     if len(cand_eps) == 0:
        #         self.sg_batch_sizes[i] = 0
        #     else:
        #         num_sg_ts = np.sum(ep_Ts)
        #         self.sg_batch_sizes[i] = (num_sg_ts / (self.current_size * (self.T-1)) * batch_size).astype(int)


        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx
