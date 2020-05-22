import numpy as np

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def achieved_cp0(achieved_goal, goal, gripper_state):
    # print(achieved_goal[2] - goal[2])
    # return goal_distance(achieved_goal,goal) < 0.04 and achieved_goal[2] - goal[2] > 0.0
    return np.linalg.norm(achieved_goal[:2] - goal[:2], axis=-1) < 0.02 and -0.1 < achieved_goal[2] - goal[2] < 0.015#0.075

#ag is gripper pos
#g is obj pos - [0,0,.025]
def achieved_cp1(achieved_goal, goal, gripper_state):
    # print(goal_distance(achieved_goal,goal), self.distance_threshold*.3, gripper_state)
    # return goal_distance(achieved_goal,goal) < self.distance_threshold
    # object_pos = goal + [0,0,0.025]
    # return achieved_goal[1] + gripper_state[0] > object_pos[1] + 0.015 and achieved_goal[1] - gripper_state[1] < object_pos[1] - 0.015 and sum(gripper_state) < 0.052 and achieved_goal[2] - object_pos[2] < .025 and -0.015 < achieved_goal[0] - goal[0] < 0.015
    # return goal_distance(achieved_goal,goal) < self.distance_threshold*.3 and gripper_state > 0.052
    # print(achieved_goal[2] - goal[2])
    return goal_distance(achieved_goal,goal) < 0.015 and gripper_state > 0.052# and achieved_goal[2] - goal[2] > 0.005
    # return achieved_sg1(achieved_goal,goal) and achieved_goal[2] - goal[2] < 0.025# and gripper_width < 0.052

#if using this have to make sure subgoal sampling is implemented, or else will say sg3 is achieved when it really isnt
def achieved_cp2(achieved_goal,goal, gripper_state):
    return goal_distance(achieved_goal,goal) < 0.05

def sample_goal_cp_0(env):
    return env.sim.data.get_site_xpos('object0').copy() + [0,0,0.025]

def sample_goal_cp_1(env):
    return env.sim.data.get_site_xpos('object0').copy()

def sample_goal_cp_2(env):
    goal = env.initial_gripper_xpos[:3] + env.np_random.uniform(-env.target_range, env.target_range, size=3)
    goal += env.target_offset
    goal[2] = env.height_offset
    if env.target_in_the_air and env.np_random.uniform() < 0.5:
        goal[2] += env.np_random.uniform(0,0.45)
    return goal

def achieved_goal_cp_0(env):
    return env.sim.data.get_site_xpos('robot0:grip').copy()

def achieved_goal_cp_1(env):
    return env.sim.data.get_site_xpos('robot0:grip').copy()

def achieved_goal_cp_2(env):
    return np.squeeze(env.sim.data.get_site_xpos('object0').copy())


checkpoints = {
    checkpoint_0 : {
        "achieved_cp" : achieved_cp0, 
        "sample_cp" : sample_goal_cp_0,
        "achieved_goal" : achieved_goal_cp_0
        },
    checkpoint_1 : {
        "achieved_cp" : achieved_cp1, 
        "sample_cp" : sample_goal_cp_1,
        "achieved_goal" : achieved_goal_cp_1
        },
    checkpoint_2 : {
        "achieved_cp" : achieved_cp2, 
        "sample_cp" : sample_goal_cp_2,
        "achieved_goal" : achieved_goal_cp_2
        }
}