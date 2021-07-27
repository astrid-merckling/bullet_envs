import numpy as np
import gym
from pybullet_data import getDataPath
import os, inspect
# import bullet_envs

from SRL4RL.utils.utilsEnv import env_with_goals

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
env_dir = os.path.abspath(os.path.join(currentdir, os.pardir))

debug = False

offset = [0, 0, 0]
_RANDOM_INIT = True

_CONTROL_VELOCITY = CONTROL_POSITION = False
CONTROL_POSITION = True





class PybulletEnv(gym.Wrapper):
    def __init__(self, env_name, urdf_root=getDataPath(), renders=False, distractor=False, noisyObs=False,
                 actionRepeat=1,  maxSteps=100,
                 image_size=84, color=False, fpv=False, noise_type='none', seed=0,doneAlive=True,
                 randomExplor=True,random_target=True,target_pos=None, display_target=False):
        if env_name == 'ReacherBulletEnv-v0':
            env = gym.make(env_name, render=renders, doneAlive=doneAlive, actionRepeat=actionRepeat,
                       randomExplor=randomExplor,distractor=distractor,random_target=random_target,
                           target_pos=target_pos,display_target=display_target, seed=seed)
        else:
            env = gym.make(env_name, render=renders, doneAlive=doneAlive, actionRepeat=actionRepeat,
                           randomExplor=randomExplor, distractor=distractor, seed=seed)
        gym.Wrapper.__init__(self, env)
        self.renders = renders
        self.deltaDiscrete = 1
        self.GoalWrapper = True
        self.backward = True
        self.maxSteps = maxSteps
        self.actionRepeat = actionRepeat
        self.maxForce = 1
        # World
        self.gravity = -10
        self.numSolverIterations = 150
        self._timeStep = 0.01
        self.distractor = distractor
        self.noisyObs = noisyObs
        self.envStepCounter = 0
        self._urdf_root = urdf_root + '/'
        self.walls = None

        # Camera
        self.fpv = fpv
        self.image_size = image_size
        self.color = color
        self.noise_type = noise_type  # possible: random_noise; random_image
        self.im_counts = 0
        self.seed(seed)

        self.reset()
        # Spaces
        observationDim = len(self.getExtendedObservation())

        self.action_space = gym.spaces.Box(low=-1., high=1, shape=(env.action_space.shape[0],),
                                           dtype=np.float32)  # self.robot.action_space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observationDim,),
                                                dtype=np.float32)  # self.robot.observation_space
        "To have the same results among different resets"
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

        self.env.env.configureCamera(self, image_size=self.image_size, noise_type=self.noise_type, color=self.color)
        self.rewardFactor = actionRepeat if env_name != 'ReacherBulletEnv-v0' else 1

    @property
    def actionRepeat(self):
        return self._actionRepeat

    @actionRepeat.setter
    def actionRepeat(self, val):
        self._actionRepeat = val

    @property
    def random_target(self):
        return self._random_target

    @random_target.setter
    def random_target(self, val):
        self._random_target = val

    @property
    def distractor(self):
        return self._distractor

    @distractor.setter
    def distractor(self, val):
        self._distractor = val

    @property
    def maxSteps(self):
        return self._maxSteps

    @maxSteps.setter
    def maxSteps(self, val):
        self._maxSteps = val

    def seed(self, seed=None):
        seed = self.env.seed(seed)[0]
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return seed
        # if seed is not None:
        #     self.seedValue = seed
        #     self.np_random = np.random
        #     self.np_random.seed(seed)
        #     self.random = random
        #     self.random.seed(seed)
        #     self.env.seed(seed)

    def render(self, mode='rgb_array', image_size=None, color=None, close=False, camera_id=0, fpv=False,
               downscaling=True):
        rgb_array = self.env.render(mode=mode, image_size=image_size, color=color, camera_id=camera_id,
                                    downscaling=downscaling)
        return rgb_array

    def reset(self):
        self.envStepCounter = 0
        self.terminated = False
        obs = self.env.reset()

        return obs

    def resetPosition(self, pos, theta=None):
        if theta is None:
            self.theta = 0
            orientation = [0, 0, 0, 1]
        else:
            self.theta = theta
            orientation = self.env.env._p.getQuaternionFromEuler([0, 0, theta])
        self.env.env._p.resetBasePositionAndOrientation(self.turtle, pos, orientation)
        self.robot_pos = np.array(self.env.env._p.getBasePositionAndOrientation(self.turtle)[0])

    def step(self, action):
        for _ in range(self.actionRepeat):
            obs, r, done, info = self.env.step(action)
            self.envStepCounter += 1
        # if self.env.robot.__class__.__name__ == 'InvertedPendulumSwingup':
        done = self._termination() or info['not alive']  # not "or done" because modified in /gym/wrappers/time_limit.py
        reward = r * self.rewardFactor
        return obs, reward, done, info

    def get_info(self):
        return {}

    def _termination(self):
        if self.terminated or self.envStepCounter >= self._maxSteps:
            return True
        return False

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def reward_reach(self, goal_a, goal_b):
        return - (self.goal_distance(goal_a, goal_b) > self.target_radius).astype(np.float32)

    def _reward(self):
        # Distance to target
        if self.with_target:
            reward = - (self.goal_distance(self.object, self.target) > self.target_radius).astype(np.float32)
        else:
            reward = 0
        return reward

    def measureRobot(self):
        return self.env.robot.measure_robot()

    def getExtendedObservation(self):
        # observation = list(self.get_info()['labels'].values())
        if self.env.env.__class__.__name__ == 'CartPoleContinuousBulletEnv':
            observation = np.array(
                self.env.env._p.getJointState(self.cartpole, 1)[0:2] + self.env.env._p.getJointState(self.cartpole, 0)[
                                                                       0:2])
        else:
            observation = self.object
        # observation.extend([np.cos(labels[-1]), np.sin(labels[-1])])
        if not self.GoalWrapper and self._random_target:
            observation.extend(list(self.target))
        return np.array((observation))

    @property
    def object(self):
        if self.env.env.__class__.__name__ + '-v0' in env_with_goals:
            self._object = self.env.robot.calc_object()
        else:
            self._object = self.env.robot.calc_state()
        return self._object


    @property
    def target(self):
        # Return only the [x, y] coordinates
        self._target = self.env.robot.calc_target()
        return self._target

    @property
    def target_radius(self):
        if self.env.env.__class__.__name__ == 'ReacherBulletEnv':
            radius = 0.015
        return radius + radius / 2.

# if env_with_goals