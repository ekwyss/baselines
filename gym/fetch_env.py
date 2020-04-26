import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, num_goals, subgoal_rewards, use_g_ind
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        # self.distance_threshold = distance_threshold #.05
        # self.distance_thresholds = [distance_threshold,distance_threshold*0.3,distance_threshold] #*0.8
        self.reward_type = reward_type
        self.num_goals = num_goals
        self.subgoal_rewards = subgoal_rewards
        self.use_g_ind = use_g_ind

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # def compute_reward(self, achieved_goal, goal, info):
    #         # if self.reward_type == 'sparse':
    #         if np.isscalar(info['is_success']):
    #             return -1.0 if info['is_success'] == 0.0 else -0.0 if info['goal_reached'] == True else self.subgoal_rewards[int(info['goal_index'])].astype(np.float32)
    #             # return -1.0 if info['is_success'] == 0.0 else -0.0
    #             # if info['goal_reached'] == False:
    #             #     rew = self.subgoal_rewards[int(info['goal_index'])].astype(np.float32)
    #             # return rew
    #         else:
    #             d = goal_distance(achieved_goal, goal)
    #             rews = np.zeros(len(info['is_success']), dtype=np.float32)
    #             for i in range(len(rews)):
    #                 if info['goal_index'][i] == 0 and sum(info['gripper_pos'][i]) < 0.52:
    #                     rews[i] = -1.0
    #                 # else:
    #                 #SGREW IN EXPERIENCE REPLAY IF ORIGINALLY FIRST TIME REACHING GOAL IN EPISODE
    #                 # rews[i] = -1.0 if info['is_success'][i] == 0.0 else -0.0 if info['goal_reached'][i] == True else self.subgoal_rewards[int(info['goal_index'][i])].astype(np.float32)
    #                 #SGREW ALWAYS IN EXPERIENCE REPLAY
    #                 # rews[i] = self.subgoal_rewards[int(info['goal_index'][i])].astype(np.float32) if info['is_success'][i] == 1 else -1.0
    #                 #NO SGREW IN EXPERIENCE REPLAY
    #                 rews[i] = -(d[i] > self.distance_threshold).astype(np.float32)
    #             # return = -(d > self.distance_threshold).astype(np.float32)
    #             # print(rews)
    #             # return rews

    #     # if self.use_g_ind == False:
    #     #     return -(d > self.distance_threshold).astype(np.float32)


    # ----------------------------
        # height_off = gripper_pos[2] - object_pos[2]
        # horiz_off = np.linalg.norm(gripper_pos[:2] - object_pos[:2], axis=-1)
        # gripper_width = sum(end_effector_pos)

        # if horiz_off < 0.075 and height_off < 0.075:
        #     if gripper_pos[1] - end_effector_pos[0] < object_pos[1] - 0.015 and gripper_pos[1] + end_effector_pos[1] > object_pos[1] + 0.015 and sum(end_effector_pos) < 0.052 and height_off < .02 and -0.01 < gripper_pos[0] - object_pos[0] < 0.01:
        #         return 2
        #     return 1
        # return 0

    # ----------------------------MODSGATT2-----------------------------
    # #ag is gripper pos
    # #g is obj pos + [0,0,.025]
    # def achieved_sg0(self, achieved_goal, goal, gripper_state):
    #     # print(achieved_goal[2] - goal[2])
    #     return goal_distance(achieved_goal,goal) < 0.04 and achieved_goal[2] - goal[2] > 0.0
    #     # return np.linalg.norm(achieved_goal[:2] - goal[:2], axis=-1) < 0.02 and -0.1 < achieved_goal[2] - goal[2] < 0.015#0.075

    # #ag is gripper pos
    # #g is obj pos - [0,0,.025]
    # def achieved_sg1(self, achieved_goal, goal, gripper_state):
    #     # print(goal_distance(achieved_goal,goal), self.distance_threshold*.3, gripper_state)
    #     # return goal_distance(achieved_goal,goal) < self.distance_threshold
    #     # object_pos = goal + [0,0,0.025]
    #     # return achieved_goal[1] + gripper_state[0] > object_pos[1] + 0.015 and achieved_goal[1] - gripper_state[1] < object_pos[1] - 0.015 and sum(gripper_state) < 0.052 and achieved_goal[2] - object_pos[2] < .025 and -0.015 < achieved_goal[0] - goal[0] < 0.015
    #     # return goal_distance(achieved_goal,goal) < self.distance_threshold*.3 and gripper_state > 0.052
    #     # print(achieved_goal[2] - goal[2])
    #     return goal_distance(achieved_goal,goal) < 0.015 and gripper_state > 0.052 and achieved_goal[2] - goal[2] > 0.005 and -0.015 < achieved_goal[0] - goal[0] < 0.015
    #     # return achieved_sg1(achieved_goal,goal) and achieved_goal[2] - goal[2] < 0.025# and gripper_width < 0.052

    # #if using this have to make sure subgoal sampling is implemented, or else will say sg3 is achieved when it really isnt
    # def achieved_sg2(self,achieved_goal,goal, gripper_state):
    #     return goal_distance(achieved_goal,goal) < 0.05

    #AND sg2 changed to 0.04
    # ----------------------------


    #ag is gripper pos
    #g is obj pos + [0,0,.025]
    def achieved_sg0(self, achieved_goal, goal, gripper_state):
        # print(achieved_goal[2] - goal[2])
        # return goal_distance(achieved_goal,goal) < 0.04 and achieved_goal[2] - goal[2] > 0.0
        return np.linalg.norm(achieved_goal[:2] - goal[:2], axis=-1) < 0.02 and -0.1 < achieved_goal[2] - goal[2] < 0.015#0.075

    #ag is gripper pos
    #g is obj pos - [0,0,.025]
    def achieved_sg1(self, achieved_goal, goal, gripper_state):
        # print(goal_distance(achieved_goal,goal), self.distance_threshold*.3, gripper_state)
        # return goal_distance(achieved_goal,goal) < self.distance_threshold
        # object_pos = goal + [0,0,0.025]
        # return achieved_goal[1] + gripper_state[0] > object_pos[1] + 0.015 and achieved_goal[1] - gripper_state[1] < object_pos[1] - 0.015 and sum(gripper_state) < 0.052 and achieved_goal[2] - object_pos[2] < .025 and -0.015 < achieved_goal[0] - goal[0] < 0.015
        # return goal_distance(achieved_goal,goal) < self.distance_threshold*.3 and gripper_state > 0.052
        # print(achieved_goal[2] - goal[2])
        return goal_distance(achieved_goal,goal) < 0.015 and gripper_state > 0.052# and achieved_goal[2] - goal[2] > 0.005
        # return achieved_sg1(achieved_goal,goal) and achieved_goal[2] - goal[2] < 0.025# and gripper_width < 0.052

    #if using this have to make sure subgoal sampling is implemented, or else will say sg3 is achieved when it really isnt
    def achieved_sg2(self,achieved_goal,goal, gripper_state):
        return goal_distance(achieved_goal,goal) < 0.05

    #only used for ER now (if planning by sg)
    def compute_reward(self, achieved_goal, goal, info):
        if np.isscalar(info['is_success']):
            # success = getattr(self, 'achieved_sg{}'.format(int(info['goal_index'])))(achieved_goal,goal, info['gripper_width'])
            #if last sg reward is 0 anyway doesn't matter if goal reached or not
            # return -1.0 if success == False  else self.subgoal_rewards[info['goal_index']]
            return -1.0 if info['is_success'] == 0  else self.subgoal_rewards[info['goal_index']]
            # return -goal_distance(achieved_goal,goal) if info['is_success'] == 0  else self.subgoal_rewards[info['goal_index']]
        else:
            rews = np.zeros(len(achieved_goal), dtype=np.float32)
            for i in range(len(rews)):
                success = getattr(self, 'achieved_sg{}'.format(int(info['goal_index'][i][0])))(achieved_goal[i],goal[i], info['gripper_width'][i])
                # rews[i] = self.subgoal_rewards[int(info['goal_index'][i][0])] if success else -1.0
                # rews[i] = -0.0 if success else -1.0
                # rews[i] = -0.0 if success else -goal_distance(achieved_goal[i],goal[i])
                rews[i] = self.subgoal_rewards[int(info['goal_index'][i][0])] if success else -1.0#goal_distance(achieved_goal[i],goal[i])
            # print(rews)
            return rews

        #######################################################################################
        ####CANT DO THIS FOR ER, IS_SUCCESS WILL HAVE NO RELEVANCE TOWARDS SUBSTITUTED GOAL####
        #######################################################################################
        # # print(info)
        # # print(info['is_success'])
        # if self.reward_type == 'sparse':
        #     if np.isscalar(info['is_success']):
        #         return -1.0 if info['is_success'] == 0.0 else -0.0 if info['goal_reached'] == True else self.subgoal_rewards[int(info['goal_index'])].astype(np.float32)
        #         # if info['goal_reached'] == False:
        #         #     rew = self.subgoal_rewards[int(info['goal_index'])].astype(np.float32)
        #         # return rew
        #     else:
        #         rews = np.zeros(len(info['is_success']), dtype=np.float32)
        #         for i in range(len(rews)):
        #             #SGREW IN EXPERIENCE REPLAY IF ORIGINALLY FIRST TIME REACHING GOAL IN EPISODE
        #             # rews[i] = -1.0 if info['is_success'][i] == 0.0 else -0.0 if info['goal_reached'][i] == True else self.subgoal_rewards[int(info['goal_index'][i])].astype(np.float32)
        #             #SGREW ALWAYS IN EXPERIENCE REPLAY
        #             rews[i] = self.subgoal_rewards[int(info['goal_index'][i])].astype(np.float32) if info['is_success'][i] == 1 else -1.0
        #             #NO SGREW IN EXPERIENCE REPLAY
        #             # rews[i] = -1.0 if info['is_success'] == 0.0 else -0.0
        #         # print(rews)
        #         return rews

        # d = goal_distance(achieved_goal, goal)
        # if self.use_g_ind == False:
        #     return -(d > self.distance_threshold).astype(np.float32)

        # if self.reward_type == 'sparse':
        #     if np.isscalar(d):
        #         # hardcode necessity for subgoal 2 gripper constraint
        #         if (info['goal_index']) == 1:
        #             if info['gripper_width'] > 0.052:
        #                 return -1.0

        #         # SG REW IN REALTIME SIMULATION
        #         if info['goal_reached'] == False:
        #             return self.subgoal_rewards[int(info['goal_index'])].astype(np.float32) if d <= self.distance_threshold else -1.0
        #         else:
        #             return -(d > self.distance_threshold).astype(np.float32)

        #         # NO SG REW IN REALTIME SIMULATION
        #         # return -(d > self.distance_threshold).astype(np.float32)
        #     else:
        #         # EXPERIENCE REPLAY
        #         rews = np.zeros(len(d), dtype=np.float32)
        #         for i in range(len(rews)):
        #             # hardcode necessity for subgoal 2 gripper constraint
        #             if info['goal_index'][i] == 1 and info['gripper_width'][i] > 0.052:
        #                 rews[i] = -1.0
        #                 continue
        #             #SGREW IN EXPERIENCE REPLAY
        #             # rews[i] = -(d[i] <= self.distance_threshold).astype(np.float32) if info['goal_reached'][i] == True else (self.subgoal_rewards[int(info['goal_index'][i])].astype(np.float32) if d[i] <= self.distance_threshold else -1.0)
        #             rews[i] = (self.subgoal_rewards[int(info['goal_index'][i])].astype(np.float32) if d[i] <= self.distance_threshold else -1.0)
        #             #NO SGREW IN EXPERIENCE REPLAY
        #             # rews[i] = -(d[i] <= self.distance_threshold).astype(np.float32)
        #         return rews
        # else:
        #     return -d

        # return -(d > self.distance_threshold).astype(np.float32)


    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        # print(action)
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if (not self.has_object) or self.goal_index < len(self.goals)-1:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])
        # print("gripper_state: ",gripper_state)
        # print(obs.shape)
        if self.use_g_ind == True:
            obs = np.concatenate([obs,[self.goal_index]])
        # print(obs.shape)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            # 'goal_index': self.goal_index,
            # 'desired_goals': self.goals.copy(),
            # 'desired_goal': self.goals.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')

        goal_pos = self.goals[self.goal_index].copy()
        self.sim.model.site_pos[site_id] = goal_pos - sites_offset[0]

        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    # def _sample_subgoal(self, sg_ind, object_pos=None):
    #     if object_pos is None:
    #         object_pos = self.sim.data.get_site_xpos('object0')

    #     #3 subgoals:
    #     if sg_ind == 0:
    #         sg0 = object_pos.copy()
    #         sg0[2] += 0.025
    #         return sg0
    #     elif sg_ind == 1:
    #         sg1 = object_pos.copy()
    #         # sg1[2] -= 0.025
    #         return sg1
    #     else:
    #         print("Goal index {} is either static or out of range")
    #         return 0

    #probably not worth it, not too big of a deal - could cause issues if calling sample_goals for some other reason
    #Don't sample first two subgoals, only sample end goal and set goals to [empty,empty,goal]
    def _sample_goals(self):
        goals = []
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            grip_pos = self.sim.data.get_site_xpos('robot0:grip')

            #preset subgoals
            if self.num_goals > 1:
                subgoal1 = object_pos.copy()
                height_off = 0.01 if self.num_goals == 2 else 0.025#0.04
                subgoal1[2] += height_off
                goals.append(subgoal1)
            #TODO: incorporate clasp object for goal 2
            if self.num_goals == 3:
                subgoal2 = object_pos.copy()
                # subgoal2[2] -= 0.025
                # goal2[2] -= 0.01
                goals.append(subgoal2)

            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        goals.append(goal.copy())
        return np.asarray(goals)

    # def _sample_goal(self):
    #     if self.has_object:
    #         goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
    #         goal += self.target_offset
    #         goal[2] = self.height_offset
    #         if self.target_in_the_air and self.np_random.uniform() < 0.5:
    #             goal[2] += self.np_random.uniform(0, 0.45)
    #     else:
    #         goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
    #     return goal.copy()

    def _is_success(self, achieved_goal, desired_goal,goal_index,gripper_width):
        return float(getattr(self, 'achieved_sg{}'.format(int(goal_index)))(achieved_goal,desired_goal, gripper_width))

        # d = goal_distance(achieved_goal, desired_goal)
        # # print(goal_index, d)
        # # hardcode necessity for subgoal 2 gripper constraint
        # # if goal_index == 1:
        # if goal_index == 0:
        #     gripper_width = self.sim.data.get_joint_qpos('robot0:l_gripper_finger_joint') + self.sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')
        #     if gripper_width < 0.052:
        #         return 0.0
        # return (d < self.distance_threshold).astype(np.float32)

        # return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    # def get_current_subgoal(self, gripper_pos, object_pos, end_effector_pos):
    #     #SIMPLE SG ATTEMPT



    #     #COMPLEX SG ATTEMPT
    #     height_off = gripper_pos[2] - object_pos[2]
    #     horiz_off = np.linalg.norm(gripper_pos[:2] - object_pos[:2], axis=-1)
    #     gripper_width = sum(end_effector_pos)
    #     print(height_off)
    #     print(horiz_off)
    #     print(np.linalg.norm(gripper_pos - object_pos, axis=-1))
    #     print(np.linalg.norm(np.array([horiz_off,height_off]), axis=-1))
    #     # print(object_pos)
    #     # print("height", height_off)
    #     # print("horiz", horiz_off)
    #     # print("width", gripper_width)
    #     # print(gripper_pos[1] - end_effector_pos[0],object_pos[1] - 0.02, object_pos[1] + 0.02,gripper_pos[1] + end_effector_pos[1])
    #     # print(gripper_pos[0] - object_pos[0])

    #     # In zone around object
    #     # if horiz_off < 0.075 and height_off < 0.075:
    #     # if horiz_off < 0.02 and height_off < 0.04:
    #     if horiz_off < 0.02 and 0.0 < height_off < 0.04:
    #         # Holding object
    #         #if object boundaries between end effectors, end effector width small enough to grasp object and 
    #         # print(gripper_pos[1], object_pos[1], end_effector_pos)
    #         # print(gripper_pos[1] + end_effector_pos[0], object_pos[1] + 0.015, object_pos[1] - 0.015, gripper_pos[1] - end_effector_pos[1])
    #         # if gripper_pos[1] + end_effector_pos[0] > object_pos[1] + 0.015 and gripper_pos[1] - end_effector_pos[1] < object_pos[1] - 0.015:
    #         #     print("between")
    #         # if sum(end_effector_pos) < 0.052:
    #         #     print("grasping")
    #         # if height_off < .02:
    #         #     print("low enough")
    #         # if -0.01 < gripper_pos[0] - object_pos[0] < 0.01:
    #         #     print("in line")
    #         if gripper_pos[1] + end_effector_pos[0] > object_pos[1] + 0.015 and gripper_pos[1] - end_effector_pos[1] < object_pos[1] - 0.015 and sum(end_effector_pos) < 0.052 and height_off < .025 and -0.015 < gripper_pos[0] - object_pos[0] < 0.015:
    #         # if gripper_pos[1] + end_effector_pos[0] > object_pos[1] + 0.015 and gripper_pos[1] - end_effector_pos[1] < object_pos[1] - 0.015 and sum(end_effector_pos) < 0.052 and height_off < .025 and -0.015 < gripper_pos[0] - object_pos[0] < 0.015:
    #         # if gripper_pos[1] + end_effector_pos[0] > object_pos[1] + 0.015 and gripper_pos[1] - end_effector_pos[1] < object_pos[1] - 0.015 and sum(end_effector_pos) < 0.052 and height_off < .02 and -0.01 < gripper_pos[0] - object_pos[0] < 0.01:
    #         # if gripper_width < 0.052 and horiz_off < 0.001 and height_off < 0.001:
    #             # print("goal 2")
    #             return 2
    #         # print("goal 1")
    #         return 1
    #     # else:
    #     # print("goal 0")
    #     return 0
    #     # return 0

    # # def holding_object(self, gripper_pos, object_pos, end_effector_pos):
    # #     return gripper_pos[1] - end_effector_pos[0] < object_pos[1] - 0.02 and gripper_pos[1] + end_effector_pos[1] > object_pos[1] + 0.02 and sum(end_effector_pos) < 0.052 and 


    # def _update_goal_status(self, obs):
    #     # action/pos coords = [forward/backward,side/side,up/down,gripper]
    #     # print(obs)
    #     gripper_pos = obs['observation'][:3]
    #     object_pos = obs['observation'][3:6]
    #     end_effector_pos = obs['observation'][9:11]
    #     # gripper_width = sum(obs['observation'][9:11])
    #     # print(gripper_pos)
    #     # print(obs['observation'][9])
    #     # print(obs['observation'][10])
    #     # print(self.goals)
    #     # print(object_pos)


    #     ag = obs['achieved_goal']
    #     g = obs['desired_goal']
 
    #     #Get current subgoal we should want to achieve given post-action observation
    #     curr_sg = self.get_current_subgoal(gripper_pos,object_pos,end_effector_pos)
    #     # print(obs)
    #     # print(obs['achieved_goal'])
    #     # print(obs['desired_goal'])
    #     # print(self.goal_index, curr_sg, getattr(self, 'achieved_sg{}'.format(self.goal_index))(obs['achieved_goal'],obs['desired_goal'], end_effector_pos))

    #     # #If current subgoal is less than goal index we know we backtracked, no success or reward
    #     if curr_sg < self.goal_index:
    #         # print("1")
    #         self.goal_index = curr_sg
    #         self.goals[curr_sg] = self._sample_subgoal(curr_sg, object_pos)
    #         self.goal = self.goals[self.goal_index]
    #         return 0.0, -1.0

    #     # #if current subgoal is same as goal index, we know we're on the same track
    #     if curr_sg == self.goal_index:
    #         # print("2")
    #         if curr_sg == self.num_goals-1:
    #             # print(obs['achieved_goal'])
    #             # print(obs['desired_goal'])
    #             d = goal_distance(obs['achieved_goal'], obs['desired_goal'])
    #             if d < self.distance_threshold:
    #                 rew = self.subgoal_rewards[self.num_goals-1] if self.goals_reached < self.num_goals else -0.0
    #                 self.goals_reached = 3
    #                 return 1.0, rew
    #             return 0.0, -1.0
    #         self.goals[curr_sg] = self._sample_subgoal(curr_sg, object_pos)
    #         self.goal = self.goals[self.goal_index]
    #         return 0.0, -1.0

    #     # if current subgoal is greater than goal index, we have achieved a subgoal
    #     if curr_sg > self.goal_index:
    #         # print("3")
    #         self.goal_index = curr_sg
    #         # If not on final goal, resample subgoal in case object has moved
    #         if curr_sg < self.num_goals-1:
    #             self.goals[curr_sg] = self._sample_subgoal(curr_sg, object_pos)
    #         self.goal = self.goals[self.goal_index]
    #         # First time we've reached goal
    #         if self.goals_reached < curr_sg:
    #             self.goals_reached = curr_sg
    #             return 0.0, self.subgoal_rewards[curr_sg-1]
    #         #been here before
    #         # print("often?")
    #         # return 1.0, -0.0
    #         return 0.0, -1.0

        # #SG 1/2 requirements
        # if horiz_off < 0.075 and height_off < 0.075:
        # # if np.linalg.norm(gripper_pos[0:3:2] - object_pos[0:3:2], axis=-1) < 0.05:
        #     # if height_off < 0.075:
        #     #SUBGOAL 2 if gripper is holding object
        #     if height_off < 0.015 and gripper_width < 0.052 and horiz_off < 0.01:
        #     # if height_off < 0.025 and gripper_width < 0.052:
        #         #just reached subgoal 2
        #         if self.goal_index < 2:
        #             self.goal_index = 2
        #             #first time reaching this subgoal
        #             if self.goals_reached < 2:
        #                 self.goals_reached = 2
        #                 return 1.0, self.subgoal_rewards[1]
        #             #previously reached this subgoal
        #             else:
        #                 return 1.0, -0.0
        #         #already reached subgoal 2 and can't possibly have backtracked or else would have reached goal
        #         else:
        #             #at goal point with object
        #             if goal_distance(object_pos, self.goals[2]) < self.distance_threshold:
        #                 #first time completing overall goal
        #                 if self.goals_reached < 3:
        #                     self.goals_reached = 3
        #                     return 1.0, self.subgoal_rewards[2]
        #                 #been at goal
        #                 else:
        #                     return 1.0, -0.0
        #             #Trying to achieve final goal
        #             return 0.0, -1.0
        #     #subgoal 1 if gripper is anywhere above the object between 0.01-.075M and within .075m horiz
        #     # else:
        #     #lower bound on sg1 vert_off
        #     elif height_off > 0:
        #         # Just reached subgoal 1 - resample
        #         if self.goal_index < 1:
        #             self.goals[1] = self._sample_subgoal(1, object_pos)
        #             self.goal_index = 1
        #             #First time reaching this subgoal
        #             if self.goals_reached < 1:
        #                 self.goals_reached = 1
        #                 return 1.0, self.subgoal_rewards[0]
        #             #Have previously reached this subgoal
        #             else:
        #                 return 1.0, -0.0
        #         #Already on subgoal 1 or backtrack from 2
        #         else:
        #             #Dropped the block and backtracking, resample
        #             if self.goal_index > 1:
        #                 self.goals[1] = self._sample_subgoal(1, object_pos)
        #                 self.goal_index = 1
        #             #otherwise just in the middle of subgoal 2, no reward in either case
        #             return 0.0,-1.0
        # #subgoal 0 if gripper is not above object or too high above object
        # #replace subgoal specification, updating on every step in case object is still moving from being dropped or bumped
        # #if self.goals_reached > 0: ? Could have this to avoid unnecessary comp but what if arm bumps object to the side?
        # self.goals[0] = self._sample_subgoal(0, object_pos)
        # #Always reset g_ind and give no reward
        # self.goal_index = 0
        # return 0.0,-1.0

        #instead of goal_reached keep track of goals_reached to avoid giving repeat first time rewards
        # just return rewards and compute is_success here?
        #will have to update compute_reward in ER because current specification for goal 0->1 not same as d<dist_thresh
        ##   have independent function for each subgoal given an observation? then can call each in here and in ER
        #will have to sample new goal for corresponding goal index if it goes backwards 
        #Use distance based rewards alongside lenient subgoal specifications?
        #Do we want first subgoal to be so lenient?
        #   Outside of ER we will additionally have episode reward acting in update so smoother trajectories are rewarded
        #   In ER, however, there will be nothing to prefer coming in lower
        #   Actually, in multiple policy case there will never be a point in coming in lower
        #       Figure out if we can incorporate overall episode reward somehow into even earlier policies
        #           Is there a terminal Q val for last state in policy 1? Can we incorporate Q val for corresponding state of next policy?
        # TODO: Only sample final goal initially, otherwise leave subgoals blank and fill in either when first reaching that goalind or backtracking

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)