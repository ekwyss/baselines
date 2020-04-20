# import baselines.her.experiment.config as config
import numpy as np
from baselines.common.mpi_moments import mpi_moments
from baselines.common import tf_util
from baselines.her.util import (
    convert_episode_to_batch_major)
    # import_function, store_args, flatten_grads, transitions_in_episode_batch, convert_episode_to_batch_major)
from baselines.her.replay_buffer import ReplayBuffer
from baselines.her.ddpg import DDPG
from baselines import logger

def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]

def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}

global DEMO_BUFFER #buffer for demonstrations

#New class for use in her.py to allow for multiple policies to appear and act as if one
class Policies:
    def __init__(self,num_policies, ddpg_params, num_goals=None, pi_from_gi=None, reuse=False, use_mpi=True):
        #if explicitly stated, allow for policies to be shared among multiple consecutive subgoals, else one policy per subgoal
        self.num_policies = num_policies
        self.num_goals = num_policies if num_goals is None else num_goals
        self.policies_used = 0
        #mapping from g_inds to policy_inds:
        self.pi_from_gi = np.arange(num_goals) if pi_from_gi is None else np.array(pi_from_gi)
        self.fst_sg_per_policy = [np.where(self.pi_from_gi == i)[0][0] for i in range(num_policies)]

        self.T = ddpg_params['T']
        self.num_demo = ddpg_params['num_demo']
        self.input_dims = ddpg_params['input_dims']
        self.batch_size = ddpg_params['batch_size']
        self.demo_batch_size = ddpg_params['demo_batch_size']
        self.bc_loss = ddpg_params['bc_loss']

        input_shapes = dims_to_shapes(self.input_dims)
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.input_dims['g'])
        buffer_shapes['ag'] = (self.T, self.input_dims['g'])
        buffer_size = (ddpg_params['buffer_size'] // ddpg_params['rollout_batch_size']) * ddpg_params['rollout_batch_size']
        #Replay Buffer to be shared among Policies using same env
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size,self.T,ddpg_params['sample_transitions']) # ReplayBuffer(buffer_shapes,buffer_size,params['T'],params['sample_transitions'])
        global DEMO_BUFFER
        DEMO_BUFFER = ReplayBuffer(buffer_shapes, buffer_size,self.T,ddpg_params['sample_transitions'])

        #Sample proportionate amount of transitions from each subgoal based on ratio of timesteps used on each subgoal during last episode (no way to keep track of total amount of each in replay buffer because it is randomly overwritten)
        # self.num_ts_per_sg = np.zeros(self.num_goals)
        #start off assuming every timestep spent towards first subgoal
        # self.num_ts_per_sg[0] = self.T

        self.policies = []
        for i in range(num_policies):
            #for now share params for all policies
            new_params = ddpg_params.copy()
            #scope must be differently named for each policy to keep distinct
            new_params['scope'] = ddpg_params['scope'] + str(i)
            # new_params['sample_transitions'] = params['sample_transitions'][i]
            self.policies.append(DDPG(reuse=reuse, **new_params, use_mpi=use_mpi))
            # self.policies.append(config.configure_ddpg(dims=dims.copy(), params=new_params.copy(), clip_return=clip_return))


    # def init_demo_buffer(self, demo_file):
    #   #TODO: make sure this is working as intended
    #   for policy in self.policies:
    #       policy.init_demo_buffer(demo_file)
    #   #Prob just want this bc will be same for all
    #   # self.policies[0].init_demo_buffer(demo_file)

    def split_episode_by_subgoal(self,episode):
        rollout_batch_size = episode['u'].shape[0]
        split = []
        for i in range(self.num_policies):
            ts = np.where(episode['info_goal_index'] == i)
            ts = [ts[1][np.nonzero(ts[0] == j)[0]] for j in range(rollout_batch_size)]
            ep_Ts = np.array([len(t) for t in ts])
            cand_eps = np.where(ep_Ts > 1)[0]
            num_sg_ts = np.sum(ep_Ts)
            split.append((ts,num_sg_ts,cand_eps))
        return split

    def init_demo_buffer(self, demoDataFile, update_stats=True): #function that initializes the demo buffer

        demoData = np.load(demoDataFile) #load the demonstration data from data file
        info_keys = [key.replace('info_', '') for key in self.input_dims.keys() if key.startswith('info_')]
        info_values = [np.empty((self.T - 1, 1, self.input_dims['info_' + key]), np.float32) for key in info_keys]

        demo_data_obs = demoData['obs']
        demo_data_acs = demoData['acs']
        demo_data_info = demoData['info']

        for epsd in range(self.num_demo): # we initialize the whole demo buffer at the start of the training
            obs, acts, goals, achieved_goals = [], [] ,[] ,[]
            i = 0
            for transition in range(self.T - 1):
                obs.append([demo_data_obs[epsd][transition].get('observation')])
                acts.append([demo_data_acs[epsd][transition]])
                goals.append([demo_data_obs[epsd][transition].get('desired_goal')])
                achieved_goals.append([demo_data_obs[epsd][transition].get('achieved_goal')])
                for idx, key in enumerate(info_keys):
                    info_values[idx][transition, i] = demo_data_info[epsd][transition][key]


            obs.append([demo_data_obs[epsd][self.T - 1].get('observation')])
            achieved_goals.append([demo_data_obs[epsd][self.T - 1].get('achieved_goal')])

            episode = dict(o=obs,
                           u=acts,
                           g=goals,
                           ag=achieved_goals)
            for key, value in zip(info_keys, info_values):
                episode['info_{}'.format(key)] = value

            episode = convert_episode_to_batch_major(episode)
            global DEMO_BUFFER
            DEMO_BUFFER.store_episode(episode) # create the observation dict and append them into the demonstration buffer
            logger.debug("Demo buffer size currently ", DEMO_BUFFER.get_current_size()) #print out the demonstration buffer size


            if update_stats:
                split = self.split_episode_by_subgoal(episode)
                for i,(sg_ts,num_sg_ts,cand_eps) in enumerate(split):
                    if len(cand_eps != 0):
                        self.policies[i].update_stats(episode, sg_ts, num_sg_ts, cand_eps)
                    # if num_sg_ts > 1:
                        # self.num_ts_per_sg[i] = num_sg_ts
                    # self.policies[i].update_stats(episode, sg_ts, num_sg_ts)
                    # else:
                        # self.num_ts_per_sg[i] = 0

            episode.clear()

        logger.info("Demo buffer size: ", DEMO_BUFFER.get_current_size()) #print out the demonstration buffer size

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        self.buffer.store_episode(episode_batch)

        if update_stats:
            split = self.split_episode_by_subgoal(episode_batch)
            for i,(sg_ts,num_sg_ts, cand_eps) in enumerate(split):
                # print(i, sg_ts, num_sg_ts, cand_eps)
                if len(cand_eps != 0):
                    # print("updating")
                    self.policies[i].update_stats(episode_batch, sg_ts, num_sg_ts, cand_eps)
                # if num_sg_ts != 0:
                # if num_sg_ts > 1:
                    # self.num_ts_per_sg[i] = num_sg_ts
                # self.policies[i].update_stats(episode_batch, sg_ts, num_sg_ts)
                # else:
                    # self.num_ts_per_sg[i] = 0

    # def store_episodes(self,episode):
    #     # episode_batch = episode
    #     # T = episode_batch['u'].shape[1]
    #     rollout_batch_size = episode['u'].shape[0]
    #     # # self.subgoals_achieved = max(len(episodes) - 2,self.subgoals_achieved)
    #     # if self.num_policies == 1:
    #     #   ep_Ts = [T]*rollout_batch_size
    #     #   self.policies[0].store_episode(episode,ep_Ts)
    #     # else:
    #     # ep_Ts = [[]*rollout_batch_size]

    #     # ep_Ts = [np.nonzero(episode_batch['info_goal_index'] == i) for i in range(self.num_goals)]
    #     # # other = np.split(other, [np.argmax(other[0] == i) for i in range(3)])
    #     # print(ep_Ts)
    #     # print([[np.sum(ep_Ts[i][0] == j) for i in range(3)] for j in range(rollout_batch_size)])
    #     # print(episode)
    #     for i in range(self.num_policies):
    #         ts = np.where(episode['info_goal_index'] == i)
    #         ts = [ts[1][np.nonzero(ts[0] == j)[0]] for j in range(rollout_batch_size)]
    #         num_sg_ts = np.sum([len(t) for t in ts])
    #         if num_sg_ts != 0:
    #             self.policies[i].store_episode(episode,ts,num_sg_ts)


    #     # ep_Ts = [[[] for i in range(self.num_policies)] for j in range(rollout_batch_size)]
    #     # for j in range(rollout_batch_size):
    #     #   goal_indices = np.where(episode_batch['info_is_success'][j] == 1)[0]
    #     #   num_ginds = len(goal_indices)

    #     #   if num_ginds > self.num_goals-1:
    #     #       goal_indices = np.concatenate((goal_indices[:self.num_goals-1],goal_indices[-1:]))
    #     #   else:
    #     #       goal_indices = np.concatenate((goal_indices,[T]))

    #     #   for i in range(self.num_policies):
    #     #       if self.fst_sg_per_policy[i] > num_ginds:
    #     #           ep_Ts[j][i].append(0)
    #     #           continue
    #     #       ep_Ts[j][i].append(goal_indices[self.fst_sg_per_policy[i]])
    #     # # print(ep_Ts)
    #     # for i in range(self.num_policies):
    #     #   ep_T = [ep_Ts[j][i][0] for j in range(rollout_batch_size)]
    #     #   # print(ep_T)
    #     #   num_candidate_transitions = sum(ep_T)
    #     #   # print(num_candidate_transitions)
    #     #   if num_candidate_transitions != 0:
    #     #       # self.subgoals_achieved = max(self.subgoals_achieved,self.fst_sg_per_policy[i])
    #     #       self.policies_used = max(self.policies_used,i)
    #     #       self.policies[i].store_episode(episode, ep_T)

    #         # for i in range(self.num_policies):

    #         #   ep_Ts = []
    #         #   for j in range(rollout_batch_size):
    #         #       goal_indices = np.where(episode_batch['info_is_success'][j] == 1)[0]

    #         #       if self.fst_sg_per_policy[i] > len(goal_indices):
    #         #           ep_Ts.append(0)
    #         #           continue

    #         #       #specific to this bc only 3 subgoals total, make more modular
    #         #       if len(goal_indices) > 2:
    #         #           goal_indices = np.concatenate((goal_indices[:2],goal_indices[-1:]))
    #         #       else:
    #         #           goal_indices = np.concatenate((goal_indices,[T]))
    #         #       ep_Ts.append(goal_indices[self.fst_sg_per_policy[i]])
    #         #   num_candidate_transitions = sum(ep_Ts)
    #         #   if num_candidate_transitions != 0:
    #         #       self.subgoals_achieved = max(self.subgoals_achieved,self.fst_sg_per_policy[i])
    #         #       self.policies[i].store_episode(episode, ep_Ts)

    #Split computation between all policies
    def train(self, stage=True):
        if stage:
            #split up batch size for each sg training based on number of timesteps spent trying to attain corresponding sg
            # ratios = np.array([sg_num_ts/sum(self.num_ts_per_sg) for sg_num_ts in self.num_ts_per_sg])
            # sg_batch_sizes = (ratios*self.batch_size).astype(int)
            # sg_demo_batch_sizes = (ratios*self.demo_batch_size).astype(int)
            # print(sg_batch_sizes)
            # print(sg_demo_batch_sizes)
            for i in range(self.num_policies):
                # if sg_batch_sizes[i] == 0:
                #     continue
                if self.bc_loss:
                    transitions = self.buffer.sample(self.batch_size - self.demo_batch_size, i)
                    global DEMO_BUFFER
                    transitions_demo = DEMO_BUFFER.sample(self.demo_batch_size, i)
                    if transitions is None and transitions_demo is None:
                        continue
                    elif transitions_demo is None:
                        self.policies[i].demo_batch_size = 0
                        self.policies[i].batch_size = transitions['u'].shape[0]
                        self.policies[i].train(transitions)
                        # print(i, self.batch_size, transitions['u'].shape)
                        # print(i, self.demo_batch_size, 0)
                    else:
                        if transitions is None:
                            transitions = {k:np.array([]) for k in transitions_demo.keys()}
                            self.policies[i].batch_size = 0
                        else:
                            self.policies[i].batch_size = transitions['u'].shape[0]
                        # print(i, self.batch_size, transitions['u'].shape)
                        # print(i, self.demo_batch_size, transitions_demo['u'].shape)
                        self.policies[i].demo_batch_size = transitions_demo['u'].shape[0]
                        for k, values in transitions_demo.items():
                            # print(transitions[k].tolist())
                            rolloutV = transitions[k].tolist()
                            for v in values:
                                rolloutV.append(v.tolist())
                            transitions[k] = np.array(rolloutV)
                        self.policies[i].train(transitions)
                else:
                    #here doesn't matter if transitions is none, will just train without staging batch
                    transitions = self.buffer.sample(self.batch_size, i)
                    if transitions is None:
                        continue
                    self.policies[i].train(transitions)
                    # transitions = self.buffer.sample(sg_batch_sizes[i], i)
        else:
            for i in range(self.num_policies):
                self.policies[i].train()

        # for i in range(self.num_policies):
        #     # if self.subgoals_achieved >= self.fst_sg_per_policy[i]:
        #     if self.policies_used >= i:
        #         self.policies[i].train()


    def update_target_nets(self):
        for i in range(self.num_policies):
            # if self.subgoals_achieved >= self.fst_sg_per_policy[i]:
            # if self.subgoals_achieved >= self.pi_from_gi[i]:
            # if self.policies_used >= i:
                # self.policies[i].update_target_net()
            self.policies[i].update_target_net()

    def record_logs(self, logger):
        prefixes = ['p' + str(i) for i in range(self.num_policies)]
        for i in range(len(self.policies)):
            for key,val in self.policies[i].logs(prefixes[i]):
                logger.record_tabular(key, mpi_average(val))

    def step(self,obs,goal_index):
        # actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goals'],goal_index)
        actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'],goal_index)
        return actions, None, None, None

    # def get_actions(self, o, ag, g, g_ind, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
    def get_actions(self, o, ag, gs, g_inds, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
        # print("o", o)
        # print("ag", ag)
        # print("gs",gs)
        # print("g_inds",g_inds)
        policy_inds = np.array([self.pi_from_gi[g_ind] for g_ind in g_inds])
        # print("policy_inds",policy_inds)
        # print(g_inds)
        # print(policy_inds)
        #TODO replace with action space dim, hardcoded 4

        acts = np.zeros((len(g_inds),4))
        if compute_Q:
            acts = [acts, np.zeros((len(g_inds),1))]
        # print("init acts",acts)
        # print(o.shape, ag.shape, gs.shape)
        for i in range(self.num_policies):
            # print(i)
            inds = np.where(policy_inds == i)[0]
            # print(inds)
            p_acts = self.policies[i].get_actions(o[inds], ag[inds], gs[inds],compute_Q=compute_Q,noise_eps=noise_eps,random_eps=random_eps,use_target_net=use_target_net)
            # print(i, "inds", inds, "p_acts",p_acts)
            # print(i, "p_acts:", p_acts)
            # acts = p_acts if i == 0 else acts+p_acts
            if compute_Q:
                acts[0][inds] = p_acts[0]
                acts[1][inds] = p_acts[1]
            else:
                acts[inds] = p_acts
            # print(i, "acts:", acts)
            # print("output")
            # print(self.policies[i].get_actions(o[inds], ag[inds], gs[inds],compute_Q=compute_Q,noise_eps=noise_eps,random_eps=random_eps,use_target_net=use_target_net))
            # output += self.policies[i].get_actions(o[inds], ag[inds], gs[inds],compute_Q=compute_Q,noise_eps=noise_eps,random_eps=random_eps,use_target_net=use_target_net)

        # policy_ind = -1
        # for sg_ind in self.fst_sg_per_policy:
        #   if g_ind >= sg_ind:
        #       policy_ind += 1

        # extra layer on gs?
        # (3,) during training
        # (1,3,3) during exec (from step above, bc had desired_goals)
        # print(g.shape)
        # print(g)
        return acts
        # return self.policies[policy_ind].get_actions(o,ag,g,compute_Q=compute_Q,noise_eps=noise_eps,random_eps=random_eps,use_target_net=use_target_net)
        # return self.policies[g_ind].get_actions(o,ag,gs[0][g_ind],compute_Q=compute_Q,noise_eps=noise_eps,random_eps=random_eps,use_target_net=use_target_net)

    def save(self, save_path):
        suffixes = ['p' + str(i) for i in range(self.num_policies)]
        for suffix in suffixes:
            tf_util.save_variables(save_path + suffix)

    #how much of other class funs do we have to implement here?
    # implement load()

#   #New class for use in her.py to allow for multiple policies to appear and act as if one
# class Policies:
#   def __init__(self,num_policies,dims,params,clip_return, num_goals=None, pi_from_gi=None):
#       #if explicitly stated, allow for policies to be shared among multiple consecutive subgoals, else one policy per subgoal
#       # self.fst_sg_per_policy = np.arange(num_policies) if fst_sg_per_policy is None else fst_sg_per_policy
#       self.num_goals = num_policies if num_goals is None else num_goals
#       self.policies_used = 0
#       # self.subgoals_achieved = 0
#       # self.fst_sg_per_policy = [0,2]
#       #mapping from g_inds to policy_inds:
#       self.pi_from_gi = np.arange(num_goals) if pi_from_gi is None else np.array(pi_from_gi)
#       # self.pi_from_gi = np.zeros(num_goals) if pi_from_gi is None else np.array(pi_from_gi)
#       self.fst_sg_per_policy = [np.where(self.pi_from_gi == i)[0][0] for i in range(num_policies)]
#       # policy_ind = -1
#       # for sg_ind in self.fst_sg_per_policy:
#       #   if g_ind >= sg_ind:
#       #       policy_ind += 1

#       self.policies = []
#       for i in range(num_policies):
#           #for now share params for all policies
#           new_params = params.copy()
#           #scope must be differently named for each policy to keep distinct
#           new_params['ddpg_params']['scope'] = "ddpg" + str(i)
#           new_params['policy_index'] = i
#           # last_goal = self.fst_sg_per_policy[i+1] if (i+1 < len(self.fst_sg_per_policy)) else num_goals
#           # relevant_goals = np.arange(self.fst_sg_per_policy[i],last_goal)
#           # self.goals_per_policy.append(relevant_goals)
#           # new_params['goal_indexes'] = relevant_goals
#           self.policies.append(config.configure_ddpg(dims=dims.copy(), params=new_params.copy(), clip_return=clip_return))
#       self.num_policies = num_policies

#   def init_demo_buffer(self, demo_file):
#       #TODO: make sure this is working as intended
#       for policy in self.policies:
#           policy.init_demo_buffer(demo_file)
#       #Prob just want this bc will be same for all
#       # self.policies[0].init_demo_buffer(demo_file)

#   def store_episodes(self,episode):
#       # episode_batch = episode
#       # T = episode_batch['u'].shape[1]
#       rollout_batch_size = episode['u'].shape[0]
#       # # self.subgoals_achieved = max(len(episodes) - 2,self.subgoals_achieved)
#       # if self.num_policies == 1:
#       #   ep_Ts = [T]*rollout_batch_size
#       #   self.policies[0].store_episode(episode,ep_Ts)
#       # else:
#       # ep_Ts = [[]*rollout_batch_size]

#       # ep_Ts = [np.nonzero(episode_batch['info_goal_index'] == i) for i in range(self.num_goals)]
#       # # other = np.split(other, [np.argmax(other[0] == i) for i in range(3)])
#       # print(ep_Ts)
#       # print([[np.sum(ep_Ts[i][0] == j) for i in range(3)] for j in range(rollout_batch_size)])
#       # print(episode)
#       for i in range(self.num_policies):
#           ts = np.where(episode['info_goal_index'] == i)
#           ts = [ts[1][np.nonzero(ts[0] == j)[0]] for j in range(rollout_batch_size)]
#           num_sg_ts = np.sum([len(t) for t in ts])
#           if num_sg_ts != 0:
#               self.policies[i].store_episode(episode,ts,num_sg_ts)


#       # ep_Ts = [[[] for i in range(self.num_policies)] for j in range(rollout_batch_size)]
#       # for j in range(rollout_batch_size):
#       #   goal_indices = np.where(episode_batch['info_is_success'][j] == 1)[0]
#       #   num_ginds = len(goal_indices)

#       #   if num_ginds > self.num_goals-1:
#       #       goal_indices = np.concatenate((goal_indices[:self.num_goals-1],goal_indices[-1:]))
#       #   else:
#       #       goal_indices = np.concatenate((goal_indices,[T]))

#       #   for i in range(self.num_policies):
#       #       if self.fst_sg_per_policy[i] > num_ginds:
#       #           ep_Ts[j][i].append(0)
#       #           continue
#       #       ep_Ts[j][i].append(goal_indices[self.fst_sg_per_policy[i]])
#       # # print(ep_Ts)
#       # for i in range(self.num_policies):
#       #   ep_T = [ep_Ts[j][i][0] for j in range(rollout_batch_size)]
#       #   # print(ep_T)
#       #   num_candidate_transitions = sum(ep_T)
#       #   # print(num_candidate_transitions)
#       #   if num_candidate_transitions != 0:
#       #       # self.subgoals_achieved = max(self.subgoals_achieved,self.fst_sg_per_policy[i])
#       #       self.policies_used = max(self.policies_used,i)
#       #       self.policies[i].store_episode(episode, ep_T)

#           # for i in range(self.num_policies):

#           #   ep_Ts = []
#           #   for j in range(rollout_batch_size):
#           #       goal_indices = np.where(episode_batch['info_is_success'][j] == 1)[0]

#           #       if self.fst_sg_per_policy[i] > len(goal_indices):
#           #           ep_Ts.append(0)
#           #           continue

#           #       #specific to this bc only 3 subgoals total, make more modular
#           #       if len(goal_indices) > 2:
#           #           goal_indices = np.concatenate((goal_indices[:2],goal_indices[-1:]))
#           #       else:
#           #           goal_indices = np.concatenate((goal_indices,[T]))
#           #       ep_Ts.append(goal_indices[self.fst_sg_per_policy[i]])
#           #   num_candidate_transitions = sum(ep_Ts)
#           #   if num_candidate_transitions != 0:
#           #       self.subgoals_achieved = max(self.subgoals_achieved,self.fst_sg_per_policy[i])
#           #       self.policies[i].store_episode(episode, ep_Ts)

# #Split computation between all policies?
#   def train(self):
#       for i in range(self.num_policies):
#           # if self.subgoals_achieved >= self.fst_sg_per_policy[i]:
#           if self.policies_used >= i:
#               self.policies[i].train()


#   def update_target_nets(self):
#       for i in range(self.num_policies):
#           # if self.subgoals_achieved >= self.fst_sg_per_policy[i]:
#           # if self.subgoals_achieved >= self.pi_from_gi[i]:
#           if self.policies_used >= i:
#               self.policies[i].update_target_net()

#   def record_logs(self, logger):
#       prefixes = ['p' + str(i) for i in range(self.num_policies)]
#       for i in range(len(self.policies)):
#           for key,val in self.policies[i].logs(prefixes[i]):
#               logger.record_tabular(key, mpi_average(val))

#   def step(self,obs,goal_index):
#       print("goal_index: ",goal_index)
#       # actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goals'],goal_index)
#       actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'],goal_index)
#       return actions, None, None, None

#   # def get_actions(self, o, ag, g, g_ind, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
#   def get_actions(self, o, ag, gs, g_inds, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
#       policy_inds = np.array([self.pi_from_gi[g_ind] for g_ind in g_inds])
#       # print(g_inds)
#       # print(policy_inds)
#       #TODO replace with action space dim, hardcoded 4
#       acts = np.zeros((len(g_inds),4))
#       for i in range(self.num_policies):
#           # print(i)
#           inds = np.where(policy_inds == i)[0]
#           # print(inds)
#           p_acts = self.policies[i].get_actions(o[inds], ag[inds], gs[inds],compute_Q=compute_Q,noise_eps=noise_eps,random_eps=random_eps,use_target_net=use_target_net)
#           # print(i, "p_acts:", p_acts)
#           # acts = p_acts if i == 0 else acts+p_acts
#           acts[inds] = p_acts
#           # print(i, "acts:", acts)
#           # print("output")
#           # print(self.policies[i].get_actions(o[inds], ag[inds], gs[inds],compute_Q=compute_Q,noise_eps=noise_eps,random_eps=random_eps,use_target_net=use_target_net))
#           # output += self.policies[i].get_actions(o[inds], ag[inds], gs[inds],compute_Q=compute_Q,noise_eps=noise_eps,random_eps=random_eps,use_target_net=use_target_net)


#       # policy_ind = -1
#       # for sg_ind in self.fst_sg_per_policy:
#       #   if g_ind >= sg_ind:
#       #       policy_ind += 1

#       # extra layer on gs?
#       # (3,) during training
#       # (1,3,3) during exec (from step above, bc had desired_goals)
#       # print(g.shape)
#       # print(g)
#       return acts
#       # return self.policies[policy_ind].get_actions(o,ag,g,compute_Q=compute_Q,noise_eps=noise_eps,random_eps=random_eps,use_target_net=use_target_net)
#       # return self.policies[g_ind].get_actions(o,ag,gs[0][g_ind],compute_Q=compute_Q,noise_eps=noise_eps,random_eps=random_eps,use_target_net=use_target_net)

#   def save(self, save_path):
#       suffixes = ['p' + str(i) for i in range(self.num_policies)]
#       for suffix in suffixes:
#           tf_util.save_variables(save_path + suffix)

#   #how much of other class funs do we have to implement here?
#   # implement load()
