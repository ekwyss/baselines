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
		self.num_goals = num_policies if num_goals is None else num_goals
		self.policies_used = 0
		# self.subgoals_achieved = 0
		#mapping from g_inds to policy_inds:
		self.pi_from_gi = np.zeros(num_goals) if pi_from_gi is None else np.array(pi_from_gi)
		self.fst_sg_per_policy = [np.where(self.pi_from_gi == i)[0][0] for i in range(num_policies)]
		# self.gi_from_pi = [np.where(self.pi_from_gi == i)[0][0] for i in range(num_policies)]
		self.max_policy_used = 0
		self.policies = []
		for i in range(num_policies):
			#for now share params for all policies
			new_params = params.copy()
			#scope must be differently named for each policy to keep distinct
			new_params['ddpg_params']['scope'] = "ddpg" + str(i)
			new_params['policy_index'] = i
			self.policies.append(config.configure_ddpg(dims=dims.copy(), params=new_params.copy(), clip_return=clip_return))
		self.num_policies = num_policies

	def init_demo_buffer(self, demo_file):
		#TODO: make sure this is working as intended
		for policy in self.policies:
			policy.init_demo_buffer(demo_file)
		# self.policies[0].init_demo_buffer(demo_file)

	def store_episodes(self,episode):
		episode_batch = episode
		T = episode_batch['u'].shape[1]
		rollout_batch_size = episode_batch['u'].shape[0]
		if self.num_policies == 1:
			ep_Ts = [T]*rollout_batch_size
			self.policies[0].store_episode(episode,ep_Ts)
		else:
			for i in range(self.num_policies):

				ep_Ts = []
				for j in range(rollout_batch_size):
					goal_indices = np.where(episode_batch['info_is_success'][j] == 1)[0]

					if self.fst_sg_per_policy[i] > len(goal_indices):
						ep_Ts.append(0)
						continue

					#specific to this bc only 3 subgoals total, make more modular
					if len(goal_indices) > 2:
						goal_indices = np.concatenate((goal_indices[:2],goal_indices[-1:]))
					else:
						goal_indices = np.concatenate((goal_indices,[T]))
					ep_Ts.append(goal_indices[self.fst_sg_per_policy[i]])
				num_candidate_transitions = sum(ep_Ts)
				if num_candidate_transitions != 0:
					# self.subgoals_achieved = max(self.subgoals_achieved,self.fst_sg_per_policy[i])
					self.policies_used = max(self.policies_used,i)
					self.policies[i].store_episode(episode, ep_Ts)

	def train(self):
		for i in range(self.num_policies):
			if self.policies_used >= i:
			# if self.subgoals_achieved >= self.fst_sg_per_policy[i]:
				self.policies[i].train()

	def update_target_nets(self):
		for i in range(self.num_policies):
			# if self.subgoals_achieved >= self.fst_sg_per_policy[i]:
			if self.policies_used >= i:
				self.policies[i].update_target_net()

	def record_logs(self, logger):
		prefixes = ['p' + str(i) for i in range(self.num_policies)]
		for i in range(len(self.policies)):
			for key,val in self.policies[i].logs(prefixes[i]):
				logger.record_tabular(key, mpi_average(val))

	def step(self,obs,goal_index):
		actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'],goal_index)
		return actions, None, None, None

	# def get_actions(self, o, ag, g, g_ind, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
	def get_actions(self, o, ag, gs, g_inds, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
		#multi env case
		policy_inds = np.array([self.pi_from_gi[g_ind] for g_ind in g_inds])
		acts = []
		for i in range(self.num_policies):
			inds = np.where(policy_inds == i)
			p_acts = self.policies[i].get_actions(o[inds], ag[inds], gs[inds],compute_Q=compute_Q,noise_eps=noise_eps,random_eps=random_eps,use_target_net=use_target_net)
			acts = p_acts if i == 0 else acts+p_acts

		return acts

	def save(self, save_path):
		suffixes = ['p' + str(i) for i in range(self.num_policies)]
		for suffix in suffixes:
			tf_util.save_variables(save_path + suffix)

	# implement load()