import sys
from baselines.run import run
import numpy as np

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

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def achieved_sg0(achieved_goal, goal, gripper_state):
    # print(achieved_goal[2] - goal[2])
    # return goal_distance(achieved_goal,goal) < 0.04 and achieved_goal[2] - goal[2] > 0.0
    return np.linalg.norm(achieved_goal[:2] - goal[:2], axis=-1) < 0.02 and -0.1 < achieved_goal[2] - goal[2] < 0.015

#ag is gripper pos
#g is obj pos - [0,0,.025]
def achieved_sg1(achieved_goal, goal, gripper_state):
    return goal_distance(achieved_goal,goal) < 0.015 and gripper_state > 0.052# and achieved_goal[2] - goal[2] > 0.005

def achieved_sg2(achieved_goal,goal,gripper_state):
    return goal_distance(achieved_goal,goal) < 0.02

#if using this have to make sure subgoal sampling is implemented, or else will say sg3 is achieved when it really isnt
def achieved_sg3(achieved_goal,goal, gripper_state):
    return goal_distance(achieved_goal,goal) < 0.05


def sample_sg_0(env):
    return env.sim.data.get_site_xpos('object0').copy() + [0,0,0.025]

def sample_sg_1(env):
    return env.sim.data.get_site_xpos('object0').copy()

def sample_sg_2(env):
    return env.sim.data.get_site_xpos('object0').copy() + [0,0,0.025]

def sample_sg_3(env):
    goal = env.initial_gripper_xpos[:3] + env.np_random.uniform(-env.target_range, env.target_range, size=3)
    goal += env.target_offset
    goal[2] = env.height_offset
    if env.target_in_the_air and env.np_random.uniform() < 0.5:
        goal[2] += env.np_random.uniform(0,0.45)
    return goal

def achieved_goal_sg_0(env):
    return env.sim.data.get_site_xpos('robot0:grip').copy()

def achieved_goal_sg_1(env):
    return env.sim.data.get_site_xpos('robot0:grip').copy()

def achieved_goal_sg_2(env):
    return np.squeeze(env.sim.data.get_site_xpos('object0').copy())

def achieved_goal_sg_3(env):
    return np.squeeze(env.sim.data.get_site_xpos('object0').copy())

checkpoints = [Checkpoint(sample_sg_0,achieved_sg0,achieved_goal_sg_0,0,0.0),Checkpoint(sample_sg_1,achieved_sg1,achieved_goal_sg_1,1,0.0),Checkpoint(sample_sg_2,achieved_sg2,achieved_goal_sg_2,2,0.0),Checkpoint(sample_sg_3,achieved_sg3,achieved_goal_sg_3,3,0.0)]

if __name__ == '__main__':
	# checkpoints = [Checkpoint(sample_sg_0,achieved_sg0,achieved_goal_sg_0,0,0.0),Checkpoint(sample_sg_1,achieved_sg1,achieved_goal_sg_1,1,0.0),Checkpoint(sample_sg_2,achieved_sg2,achieved_goal_sg_2,2,0.0),Checkpoint(sample_sg_3,achieved_sg3,achieved_goal_sg_3,3,0.0)]
	run(sys.argv, checkpoints)