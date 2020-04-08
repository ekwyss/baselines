import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500

class RobotEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goals = self._sample_goals()
        self.goal_index = 0
        self.goal = self.goals[self.goal_index]
        self.goal_reached = False
        # self.goals_reached = 0
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            desired_goals=spaces.Box(-np.inf, np.inf, shape=obs['desired_goals'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        goal = self.goals[self.goal_index].copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        
        done = False

        # info = dict()

        # if self.use_g_ind == True:
        #     info['goal_index'] = self.goal_index
        #     info['goals_reached'] = self.goals_reached
        #     # info['goal_reached'] = self.goal_reached
        #     info['gripper_width'] = self.sim.data.get_joint_qpos('robot0:l_gripper_finger_joint') + self.sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')
        
        # # NEW: Check what subgoal you are currently on from state of environment, regardless of last goal index
        # is_success, reward = self._update_goal_status()

        # info['is_success'] = is_success

        # REPLACING WITH _UPADTE_GOAL_STATUS to handle goal_index,goal,goals_reached, and reward
        info = {
            'is_success': self._is_success(obs['achieved_goal'], goal,self.goal_index), #orig
            # 'goal_index': self.goal_index,
            # 'goal_reached': self.goal_reached
            # 'consistent_subgoals': [self._is_success(obs['achieved_goal'],goal) for goal in self.goals]
            #first two are if hand is close and last is if object is close
            # 'consistent_subgoals': [self._is_success(self.sim.data.get_site_xpos('robot0:grip'),self.goals[0]), self._is_success(self.sim.data.get_site_xpos('robot0:grip'),self.goals[1]), self._is_success(self.sim.data.get_site_xpos('object0'),self.goals[2])]
            # 'consistent_subgoals': [self._is_success(self.sim.data.get_site_xpos('robot0:grip'),self.goals[0])]
            # 'consistent_subgoals': [self._is_success(self.sim.data.get_site_xpos('robot0:grip'),self.goals[i]) for i in range(self.num_goals)]
        }

        if self.use_g_ind == True:
            info['goal_index'] = self.goal_index
            info['goal_reached'] = self.goal_reached
            info['gripper_width'] = self.sim.data.get_joint_qpos('robot0:l_gripper_finger_joint') + self.sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')
        
        reward = self.compute_reward(obs['achieved_goal'], goal, info)#self.goal

        #update goal index if reached subgoal
        # if info['is_success'] == 1 and not self.goal_reached:
        if reward != -1 and not self.goal_reached:# and self.goal_index < len(self.goals)-1: #second part was commented out
            # reward = self.subgoal_rewards[self.goal_index]
            if self.goal_index == self.num_goals-1:
                self.goal_reached = True
            if self.goal_index < self.num_goals-1:
                self.goal_index += 1
                self.goal = self.goals[self.goal_index]


        # info['goal_index'] = self.goal_index
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goals = self._sample_goals()
        self.goal_index = 0
        self.goal = self.goals[self.goal_index]
        # self.goals_reached = 0
        self.goal_reached = False
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goals(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    # def _sample_goal(self):
    #     """Samples a new goal and returns it.
    #     """
    #     raise NotImplementedError()

    # def _update_goal_state():
    #     """Updates goal_index and goal_reached based on state of the environment.
    #     This way we can jump forwards or backwards goals to act dynamically instead of just sequentially.
    #     """
    #     raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
