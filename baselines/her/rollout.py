from collections import deque

import numpy as np
import pickle

from baselines.her.util import convert_episode_to_batch_major, store_args


class RolloutWorker:

    @store_args
    def __init__(self, venv, policies, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, monitor=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            venv: vectorized gym environments.
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """

        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.reset_all_rollouts()
        self.clear_history()

    def reset_all_rollouts(self):
        self.obs_dict = self.venv.reset()
        self.initial_o = self.obs_dict['observation']
        self.initial_ag = self.obs_dict['achieved_goal']
        self.g = self.obs_dict['desired_goal']
        #remove extra layer of array
        self.gs = self.obs_dict['desired_goals']#[0]
        # self.g_index = 0
        # self.subgoal_timesteps = [[0],[0],[0]]
        # self.g = self.gs[self.g_index].copy()

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        # consistent_sgss = []
        dones = []
        info_values = [np.empty((self.T - 1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        # print("new ep")
        # g_index = 0
        g_indices = [0]*self.rollout_batch_size
        # self.policies.g_index = 0
        for t in range(self.T):
            # policy_output = self.policy.get_actions(
            #     # o, ag, self.gs[self.g_index],
            #     # o, ag, self.g,
            #     o, ag, self.gs[2],
            #     compute_Q=self.compute_Q,
            #     noise_eps=self.noise_eps if not self.exploit else 0.,
            #     random_eps=self.random_eps if not self.exploit else 0.,
            #     use_target_net=self.use_target_net)

            # print(o)
            # print(o.shape)
            # print(ag.shape)
            # print(self.gs)
            # print(self.gs.shape)
            # print(self.gs[0][g_index])
            # print(self.g)
            # print(self.gs[0][g_index].shape)

            #num_env = 2: (same with num_cpu = n)
            #2,25                       | 1,25
            #2,3                        | 1,3
            #2,3,3                      | 1,3,3
            #[1.47,.62,.45]             | [1.46,.62,.45]
            #3,                         | 3,
            # [g[i][g_inds[i]] for i in range(len(g_inds))]
            # sgs = np.array([a[b] for a,b in zip(self.gs,g_indices)])

            self.g = np.array([a[b] for a,b in zip(self.gs,g_indices)]) 
            # print(self.gs)
            # print(g_indices)
            # print(self.g)
            policy_output = self.policies.get_actions(
                # o, ag, self.gs, g_index,
                # o, ag, self.gs[0][g_index], g_index,
                # o, ag, sgs, g_indices,
                o, ag, self.g, g_indices,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            obs_dict_new, rewards, done, info = self.venv.step(u)

            #TODO: All definitely only works for one env, extend for any num
            # g_index_new = obs_dict_new['goal_index'] #make sure this doesn't change outside of this
            # consistent_sgs = info[0]['consistent_subgoals'] 
            # consistent_sgs = np.array([i.get('consistent_subgoals', 0.0) for i in info])

            o_new = obs_dict_new['observation']
            ag_new = obs_dict_new['achieved_goal']
            success = np.array([i.get('is_success', 0.0) for i in info])

            # self.g_index = g_index_new

            #update goal/goal_index if we achieve a subgoal
            for i in np.where(rewards != -1)[0]:
                # print(i)
                g_indices[i] = min(g_indices[i]+1,self.policies.num_goals-1)
                # print("?")
                # self.g = [self.gs[:,g_indices]]

            # if reward != -1 and g_index < len(self.gs[0])-1:#[0])-1:
            #     g_index += 1
            #     #would have to be of len(numenvs)
            #     self.g = [self.gs[0][g_index]]



            # #identify transition as candidate for subgoal experience replay
            # for i in range(len(consistent_sgs)):
            #     if consistent_sgs[i] == 1:
            #         self.subgoal_timesteps[i].append(t)

            # if g_index_new != self.g_index:
            #     self.subgoal_timesteps.append(t)
            #     self.g_index = g_index_new

            if any(done):
                # here we assume all environments are done is ~same number of steps, so we terminate rollouts whenever any of the envs returns done
                # trick with using vecenvs is not to add the obs from the environments that are "done", because those are already observations
                # after a reset
                break

            for i, info_dict in enumerate(info):
                for idx, key in enumerate(self.info_keys):
                    info_values[idx][t, i] = info[i][key]

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            # consistent_sgss.append(consistent_sgs.copy())
            dones.append(done)
            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            # goals.append(self.gs[self.g_index].copy())
            o[...] = o_new
            ag[...] = ag_new

            #in case subgoal was achieved
            # self.g = obs_dict_new['desired_goal'].copy()

            # if reward != -1 and self.g_index < len(self.goals):
                # self.g_index += 1

        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)#,
                       #sgt=consistent_sgss)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def save_policies(self,path):
        suffixes = ['1','2','3']
        for i in range(len(self.policies)):
            with open(path+suffixes[i],'wb') as f:
                pickle.dump(self.policies[i],f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

