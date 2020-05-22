class Checkpoint():

	def __init__(self, sample_func, achieved_func, get_ag_func, index, reward):
		self.sample_func = sample_func
		self.achieved_func = achieved_func
		self.get_ag_func = get_ag_func
		self.index = index
		self.reward = reward

	def sample_sg(self, env):
		return self.sample_func(env)

	def get_ag(self, env):
		return self.get_ag_func(env)

	def achieved_sg(self, achieved_goal, goal, info):
		return self.achieved_func(achieved_goal, goal, info)