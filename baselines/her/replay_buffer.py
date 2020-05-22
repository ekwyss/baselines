import threading

import numpy as np
import time

class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions, batch_size, num_goals):
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
        #Matrix holding timesteps of transitions attempting to achieve each subgoal within each episode
        self.Ts_IJ = []

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}
        # self.Ts_IJ = [None] * self.size
        # self.t_samples = [None] * self.size
        # self.future_offsets = [None] * self.size
        self.checkpoint_Ts = np.ones((self.size,num_goals)) * self.T-1

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.batch_size = batch_size
        # self.future_p = 0.8 #CHANGE
        self.num_goals = num_goals

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        # print("buffers", self.buffers, self.buffers.keys(), self.current_size)
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        # start = time.time()
        # transitions = self.sample_transitions(buffers, batch_size, self.Ts_IJ[:self.current_size])#, [buffers['u'].shape[1]]*buffers['u'].shape[0])
        transitions = self.sample_transitions(buffers, batch_size, self.checkpoint_Ts[:self.current_size])#, [buffers['u'].shape[1]]*buffers['u'].shape[0])
        # print("sample transitions:", time.time() - start)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        # print(batch_sizes)
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

        for i, idx in enumerate(idxs):
            for j in range(1,self.num_goals):
                checkpoint_idxs = np.where(episode_batch['info_goal_index'][i] == j)[0]
                #change to <=1 ?
                if len(checkpoint_idxs) == 0:
                    self.checkpoint_Ts[idx][j-1] = self.T-1
                else:
                    self.checkpoint_Ts[idx][j-1] = checkpoint_idxs[0]
        # print(self.checkpoint_Ts[:self.current_size])


            # print([np.where(episode_batch['info_goal_index'][i] == j)[0][0] for j in range(self.num_goals)])
        # print(10/0)


        # for i,idx in enumerate(idxs):
        #     Ts_i = []
        #     for j in range(3):
        #         Ts_ij = np.where(episode_batch['info_goal_index'][i] == j)[0]
        #         Ts_i.append(Ts_ij if len(Ts_ij) > 1 else np.array([],dtype=int))
        #     self.Ts_IJ[idx] = Ts_i

        # #experiment with amount here
        # # num_her_idxs = max(10, batch_size / self.current_size)
        # num_her_idxs = 10 + int(self.batch_size / self.current_size)
        # print(num_her_idxs)

        # for i,idx in enumerate(idxs):
        #     #J x varying
        #     Ts_J = []
        #     # 1 x j
        #     len_ts_j = np.zeros(self.num_goals)

        #     #remove stretches of episode where there's only one transition towards a subgoal
        #     for j in range(self.num_goals):
        #         Ts_ij = np.where(episode_batch['info_goal_index'][i] == j)[0]
        #         if len(Ts_ij) > 1:
        #             Ts_J.append(Ts_ij)
        #             # len_ts_j.append(len(Ts_ij))
        #             len_ts_j[j] = len(Ts_ij)
        #         else:
        #             Ts_J.append(np.array([],dtype=int))
        #             # len_ts_j.append(0)

        #     probas = len_ts_j / sum(len_ts_j)
        #     #instead of t_per_ep calc each time sampling, store some number instead to sample from
        #     # true amount sampled each time is future_p * batch_size / self.current_size (early on this is large amount, later on 0-1 per episode)
        #     sg_indices = np.random.choice(self.num_goals, num_her_idxs, p=probas)
        #     sg_Ts = [np.sum(sg_indices==i) for i in range(3)]

        #     for j in range(self.num_goals):
        #         sg_T = sg_Ts[j]
        #         #No samples (covers case if num_sg_ts == 0
        #         if sg_T == 0:
        #             continue
        #         sg_ts = Ts_J[j]
        #         num_sg_ts = len(sg_ts)

        #         #sample INDICES OF timesteps to use for ER, don't sample last timestep because want to keep ER within sg
        #         t_samps_i = np.random.randint(num_sg_ts-1, size=sg_T)# if num_sg_ts > 1 else np.zeros(sg_T).astype(int)
        #         future_offset_i = np.random.uniform(size=sg_T) * (num_sg_ts-1 - t_samps_i)
        #         future_offset_i = future_offset_i.astype(int)

        #         t_samps = sg_ts[t_samps_i]
        #         #ADDING 1 HERE AND REMOVING 1 BELOW WHEN CALC FUTURE_T, TO STAY WITHIN SG
        #         future_offset = sg_ts[future_offset_i+t_samps_i+1] - sg_ts[t_samps_i]

        #         if j == 0:
        #             self.t_samples[idx] = t_samps
        #             self.future_offsets[idx] = future_offset
        #         else:
        #             self.t_samples[idx] = np.concatenate(self.t_samples[idx], t_samps)
        #             self.future_offsets[idx] = np.concatenate(self.future_offsets[idx], future_offset)
        # print(self.Ts_IJ[:self.current_size])
        # print(self.t_samples[:self.current_size])
        # print(self.future_offsets[:self.current_size])



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
            replace = -1
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
            replace = 0
        else:
            idx = np.random.randint(0, self.size, inc)
            replace = 1

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx
