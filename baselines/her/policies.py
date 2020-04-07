import baselines.her.experiment.config as config
import numpy as np
from baselines.common.mpi_moments import mpi_moments
from baselines.common import tf_util

def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]

#New class for use in her.py to allow for multiple policies to appear and act as if one
class Policies:
	def __init__(self,num_policies,dims,params,clip_return, num_goals=None, pi_from_gi=None):
		#if explicitly stated, allow for policies to be shared among multiple consecutive subgoals, else one policy per subgoal
		# self.fst_sg_per_policy = np.arange(num_policies) if fst_sg_per_policy is None else fst_sg_per_policy
		self.num_goals = num_policies if num_goals is None else num_goals
		self.policies_used = 0
		# self.subgoals_achieved = 0
		# self.fst_sg_per_policy = [0,2]
		#mapping from g_inds to policy_inds:
		self.pi_from_gi = np.arange(num_goals) if pi_from_gi is None else np.array(pi_from_gi)
		# self.pi_from_gi = np.zeros(num_goals) if pi_from_gi is None else np.array(pi_from_gi)
		self.fst_sg_per_policy = [np.where(self.pi_from_gi == i)[0][0] for i in range(num_policies)]
		# policy_ind = -1
		# for sg_ind in self.fst_sg_per_policy:
		# 	if g_ind >= sg_ind:
		# 		policy_ind += 1

		self.policies = []
		for i in range(num_policies):
			#for now share params for all policies
			new_params = params.copy()
			#scope must be differently named for each policy to keep distinct
			new_params['ddpg_params']['scope'] = "ddpg" + str(i)
			new_params['policy_index'] = i
			# last_goal = self.fst_sg_per_policy[i+1] if (i+1 < len(self.fst_sg_per_policy)) else num_goals
			# relevant_goals = np.arange(self.fst_sg_per_policy[i],last_goal)
			# self.goals_per_policy.append(relevant_goals)
			# new_params['goal_indexes'] = relevant_goals
			self.policies.append(config.configure_ddpg(dims=dims.copy(), params=new_params.copy(), clip_return=clip_return))
		self.num_policies = num_policies

	def init_demo_buffer(self, demo_file):
		#TODO: make sure this is working as intended
		for policy in self.policies:
			policy.init_demo_buffer(demo_file)
		#Prob just want this bc will be same for all
		# self.policies[0].init_demo_buffer(demo_file)

	def store_episodes(self,episode):
		episode_batch = episode
		T = episode_batch['u'].shape[1]
		rollout_batch_size = episode_batch['u'].shape[0]
		# self.subgoals_achieved = max(len(episodes) - 2,self.subgoals_achieved)
		if self.num_policies == 1:
			ep_Ts = [T]*rollout_batch_size
			self.policies[0].store_episode(episode,ep_Ts)
		else:
			# ep_Ts = [[]*rollout_batch_size]
			ep_Ts = [[[] for i in range(self.num_policies)] for j in range(rollout_batch_size)]
			for j in range(rollout_batch_size):
				goal_indices = np.where(episode_batch['info_is_success'][j] == 1)[0]
				num_ginds = len(goal_indices)

				if num_ginds > self.num_goals-1:
					goal_indices = np.concatenate((goal_indices[:self.num_goals-1],goal_indices[-1:]))
				else:
					goal_indices = np.concatenate((goal_indices,[T]))

				for i in range(self.num_policies):
					if self.fst_sg_per_policy[i] > num_ginds:
						ep_Ts[j][i].append(0)
						continue
					ep_Ts[j][i].append(goal_indices[self.fst_sg_per_policy[i]])
			# print(ep_Ts)
			for i in range(self.num_policies):
				ep_T = [ep_Ts[j][i][0] for j in range(rollout_batch_size)]
				# print(ep_T)
				num_candidate_transitions = sum(ep_T)
				# print(num_candidate_transitions)
				if num_candidate_transitions != 0:
					# self.subgoals_achieved = max(self.subgoals_achieved,self.fst_sg_per_policy[i])
					self.policies_used = max(self.policies_used,i)
					self.policies[i].store_episode(episode, ep_T)

			# for i in range(self.num_policies):

			# 	ep_Ts = []
			# 	for j in range(rollout_batch_size):
			# 		goal_indices = np.where(episode_batch['info_is_success'][j] == 1)[0]

			# 		if self.fst_sg_per_policy[i] > len(goal_indices):
			# 			ep_Ts.append(0)
			# 			continue

			# 		#specific to this bc only 3 subgoals total, make more modular
			# 		if len(goal_indices) > 2:
			# 			goal_indices = np.concatenate((goal_indices[:2],goal_indices[-1:]))
			# 		else:
			# 			goal_indices = np.concatenate((goal_indices,[T]))
			# 		ep_Ts.append(goal_indices[self.fst_sg_per_policy[i]])
			# 	num_candidate_transitions = sum(ep_Ts)
			# 	if num_candidate_transitions != 0:
			# 		self.subgoals_achieved = max(self.subgoals_achieved,self.fst_sg_per_policy[i])
			# 		self.policies[i].store_episode(episode, ep_Ts)

#Split computation between all policies?
	def train(self):
		for i in range(self.num_policies):
			# if self.subgoals_achieved >= self.fst_sg_per_policy[i]:
			if self.policies_used >= i:
				self.policies[i].train()


	def update_target_nets(self):
		for i in range(self.num_policies):
			# if self.subgoals_achieved >= self.fst_sg_per_policy[i]:
			# if self.subgoals_achieved >= self.pi_from_gi[i]:
			if self.policies_used >= i:
				self.policies[i].update_target_net()

	def record_logs(self, logger):
		prefixes = ['p' + str(i) for i in range(self.num_policies)]
		for i in range(len(self.policies)):
			for key,val in self.policies[i].logs(prefixes[i]):
				logger.record_tabular(key, mpi_average(val))

	def step(self,obs,goal_index):
		print("goal_index: ",goal_index)
		# actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goals'],goal_index)
		actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'],goal_index)
		return actions, None, None, None

	# def get_actions(self, o, ag, g, g_ind, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
	def get_actions(self, o, ag, gs, g_inds, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
		policy_inds = np.array([self.pi_from_gi[g_ind] for g_ind in g_inds])
		# print(g_inds)
		# print(policy_inds)
		#TODO replace with action space dim, hardcoded 4
		acts = np.zeros((len(g_inds),4))
		for i in range(self.num_policies):
			# print(i)
			inds = np.where(policy_inds == i)[0]
			# print(inds)
			p_acts = self.policies[i].get_actions(o[inds], ag[inds], gs[inds],compute_Q=compute_Q,noise_eps=noise_eps,random_eps=random_eps,use_target_net=use_target_net)
			# print(i, "p_acts:", p_acts)
			# acts = p_acts if i == 0 else acts+p_acts
			acts[inds] = p_acts
			# print(i, "acts:", acts)
			# print("output")
			# print(self.policies[i].get_actions(o[inds], ag[inds], gs[inds],compute_Q=compute_Q,noise_eps=noise_eps,random_eps=random_eps,use_target_net=use_target_net))
			# output += self.policies[i].get_actions(o[inds], ag[inds], gs[inds],compute_Q=compute_Q,noise_eps=noise_eps,random_eps=random_eps,use_target_net=use_target_net)


		# policy_ind = -1
		# for sg_ind in self.fst_sg_per_policy:
		# 	if g_ind >= sg_ind:
		# 		policy_ind += 1

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