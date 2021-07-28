import pybullet
from pybullet_utils import bullet_client
import numpy as np
import cv2, os
import time
import gym
from datetime import datetime
from collections import OrderedDict
from pybullet_data import getDataPath
import random

from bullet_envs.utils import seeding_np_random, AddNoise

robot_diameter = 0.4

initZ = 0.


class TurtlebotMazeEnv(TurtlebotEnv):
    def __init__(self, urdf_root=getDataPath(), renders=False, random_target=False, target_pos=None, distractor=False,
                 actionRepeat=1, maxSteps=100,
                 image_size=84, color=False, fpv=True, display_target=False, randomExplor=True, noise_type='none',
                 with_velocity=False, with_ANGLES=False, seed=None, with_target=True,
                 wallDistractor=False, debug=False):
        assert fpv, 'TurtlebotMazeEnv-v0 only with fpv'
        super().__init__(urdf_root=urdf_root, renders=renders, random_target=random_target, target_pos=target_pos,
                         distractor=distractor, actionRepeat=actionRepeat,
                         maxSteps=maxSteps,
                         image_size=image_size, color=color, fpv=fpv, display_target=display_target,
                         randomExplor=randomExplor, noise_type=noise_type,
                         with_velocity=with_velocity, with_ANGLES=with_ANGLES, seed=seed,
                         with_target=with_target, wallDistractor=wallDistractor, debug=debug)



class TurtlebotEnv(gym.Env):
    def __init__(self, urdf_root=getDataPath(), renders=False, distractor=False,
                 actionRepeat=1, maxSteps=100,
                 image_size=84, color=False, fpv=False, randomExplor=True, noise_type='none',
                 with_velocity=False, with_ANGLES=False, seed=None,
                 with_target=True, wallDistractor=False, random_target=False, target_pos=None, display_target=False,
                 debug=False):

        # Agent
        self.script_path = os.path.dirname(__file__)
        self.class_name = self.__class__.__name__
        self.with_target = with_target
        self.display_target = display_target
        self.randomExplor = randomExplor
        self.deltaDiscrete = 1
        self.GoalWrapper = True
        self.backward = True
        self.maxSteps = maxSteps
        self.actionRepeat = actionRepeat
        self.numSolverIterations = 5
        self.timeStep = 0.0165
        self.maxForce = 1
        self.robot_diameter = robot_diameter
        self.debug = debug
        # World
        self.gravity = -10
        self.random_target = random_target
        self.distractor = distractor
        self.envStepCounter = 0
        self._urdf_root = urdf_root + '/'
        self.walls = None

        if self.class_name == 'TurtlebotMazeEnv':
            self.collision_margin = 0.25
            self.wallLimit = self.collision_margin + robot_diameter / 2
            self._min_x, self._max_x = 0., 1.50 * 4
            self._min_y, self._max_y = 0., 1.50 * 3
            self.posMultiplier = 4 * (1.50 * 3 - self.wallLimit) / 64
            self._cam_dist = self._max_x
            self.camera_target_pos = (self._max_x / 2., self._max_y / 2., 0)
        else:
            self.collision_margin = 0.2
            self.wallLimit = self.collision_margin + robot_diameter / 2
            # Boundaries of the square env
            camera_size = robot_diameter * 100 / 20
            self._min_x, self._max_x = 0., camera_size
            self._min_y, self._max_y = 0., camera_size
            self.posMultiplier = 4 * (min(self._max_x,self._max_y) - self.wallLimit) / 64  # to move 4 pixels in a 64x64 image
            self._cam_dist = 0.85 * self._max_x  # 4.2 #5.2
            self.camera_target_pos = (self._max_x / 2., self._max_y / 2., 0)

        self.speedMultiplier = 66
        # Target
        self.target_pos = target_pos
        if target_pos is not None:
            assert not random_target
            if len(self.target_pos) == 2:
                self.target_pos = np.hstack([self.target_pos, 0])
            else:
                self.target_pos = np.hstack(self.target_pos)
            print('target_pos {}'.format(target_pos))

        # Camera
        self.renders = renders
        self.fpv = fpv
        self.image_size = image_size
        self.color = color
        self.noise_type = noise_type
        self.im_counts = 0
        self.np_random, seed = seeding_np_random(seed=seed)
        self.window_w = 512
        self.window_h = 512
        self.state_w = 96  # 96
        self.state_h = 96  # 96

        if self.noise_type != 'none':
            self.noise_adder = AddNoise(self)
        "include stochastic dynamics transition"
        self.wallDistractor = wallDistractor

        self._cam_yaw = 90
        self._cam_pitch = -90
        self._cam_roll = 0
        self.renderer = pybullet.ER_BULLET_HARDWARE_OPENGL

        self.stateId = -1
        self.physicsClientId = -1
        self.walls = None
        self.turtle = None
        self.target_uid = None
        self.ownsPhysicsClient = False
        self.reset()
        # Spaces
        observationDim = len(self.getExtendedObservation())
        self.action_space = gym.spaces.Box(low=-1., high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observationDim,),
                                                dtype=np.float32)
        "To have the same results among different resets"
        self.seed(seed)

    def define_goal(self, val):
        self.new_goal = np.hstack([val, 0])
        self._p.resetBasePositionAndOrientation(self.target_uid, self.new_goal, [0, 0, 0, 1])

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
        if seed is None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            self.np_random = self.np_random
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return seed

    def close(self):
        if self.ownsPhysicsClient:
            if self.physicsClientId >= 0:
                self._p.disconnect()
        self.physicsClientId = -1

    def render(self, mode='rgb_array', image_size=None, color=None, close=False, camera_id=0,
               fpv=None, downscaling=True):

        if mode == 'rgb_array':
            if downscaling:
                VP_W = self.state_w
                VP_H = self.state_h
            else:
                VP_W = image_size
                VP_H = image_size
        else:
            VP_W = self.window_w
            VP_H = self.window_h

        image_size = self.image_size if image_size is None else image_size
        color = self.color if color is None else color
        im_shapes = [image_size, image_size, 3] if color else [image_size, image_size, 1]
        fpv = self.fpv if fpv is None else fpv

        # for 'plate' noise
        if 'plate' in self.noise_type:
            self.noise_adder(self._p, im_shapes=im_shapes)

        if camera_id == -1 and not fpv:
            "presentation view of the environment"
            if self.class_name == 'TurtlebotEnv':
                cameraTargetPosition = [self._max_x / 2. + 0.8, self._max_y / 1.4 - 2, 1.6]
                yaw = 30
                pitch = -40
            else:
                yaw = 180
                pitch = -25
                cameraTargetPosition = (self._max_x / 2., 2 * self._max_y - self._max_y / 6, 3.3)
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=cameraTargetPosition,
                distance=1,
                yaw=yaw,
                pitch=pitch,
                roll=self._cam_roll,
                upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=60, aspect=float(VP_W) / VP_H,
                nearVal=0.1, farVal=100.0)
            (_, _, px1, _, _) = self._p.getCameraImage(
                width=VP_W, height=VP_H, viewMatrix=view_matrix,
                projectionMatrix=proj_matrix, renderer=self.renderer)
        elif camera_id == 1 and not fpv:
            "map view of the environment"
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.camera_target_pos,
                distance=self._cam_dist,
                yaw=180,
                pitch=self._cam_pitch,
                roll=self._cam_roll,
                upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=60, aspect=float(VP_W) / VP_H,
                nearVal=0.1, farVal=100.0)
            (_, _, px1, _, _) = self._p.getCameraImage(
                width=VP_W, height=VP_H, viewMatrix=view_matrix,
                projectionMatrix=proj_matrix, renderer=self.renderer)
        elif fpv:
            "first-person view of the robot"
            robot_pos = self._object
            future_robot_pos = self.bump_detection(np.hstack([np.cos(self.theta), np.sin(self.theta)]), bump=False)
            delta = (future_robot_pos - robot_pos) / 1
            view_matrix = self._p.computeViewMatrix(
                cameraEyePosition=(robot_pos[0] + delta[0], robot_pos[1] + delta[1], 0.13),
                cameraTargetPosition=(future_robot_pos[0] + delta[0], future_robot_pos[1] + delta[1], 0.12),
                cameraUpVector=[0, 0, 1])
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=90, aspect=float(VP_W) / VP_H,
                nearVal=0.1, farVal=100.0)
            (_, _, px1, _, _) = self._p.getCameraImage(
                width=VP_W, height=VP_H, viewMatrix=view_matrix,
                projectionMatrix=proj_matrix, renderer=self.renderer)

        rgb_array = np.array(px1)[:, :, :3].astype(np.float32)

        if not color:
            rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
            rgb_array = np.expand_dims(rgb_array, -1)
        if self.noise_type in ['noisyObs', 'random_noise', 'random_image', 'random_cutout']:
            rgb_array = self.noise_adder(observation=rgb_array)

        if mode == "rgb_array":
            if image_size != VP_W:
                "Resize to a square image, important that rgb_array values between 0 and 255"
                rgb_array = cv2.resize(rgb_array.astype(np.uint8), dsize=(image_size, image_size),
                                       interpolation=cv2.INTER_CUBIC).astype(np.float32)
            else:
                rgb_array = rgb_array.astype(np.float32)

        return rgb_array

    def init_goal(self):
        margin = self.wallLimit
        if self.class_name == 'TurtlebotMazeEnv':
            zone = np.random.randint(1, 4)
            if zone == 1:  # left
                x_pos = np.random.uniform(0 + margin, 1.50 * 4 - margin)
                y_pos = np.random.uniform(0 + margin, 1.50 - margin)
            elif zone == 2:  # right
                x_pos = np.random.uniform(0 + margin, 1.50 * 4 - margin)
                y_pos = np.random.uniform(1.50 * 2 + margin, 1.50 * 3 - margin)
            elif zone == 3:  # top
                x_pos = np.random.uniform(0 + margin, 1.50 - margin)
                y_pos = np.random.uniform(1.50 + margin, 2 * 1.50 - margin)
        else:
            x_pos = np.random.uniform(self._min_x + margin, self._max_x - margin)
            y_pos = np.random.uniform(self._min_y + margin, self._max_y - margin)
        return np.hstack([x_pos, y_pos, 0])

    def _reset(self):
        if self.physicsClientId < 0:
            self.ownsPhysicsClient = True

            if self.renders:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
                self._p.resetDebugVisualizerCamera(3, 180, -41, [self._max_x / 2. + 1, self._max_y / 1.4, 1.])
            else:
                self._p = bullet_client.BulletClient()

            self.physicsClientId = self._p._client
            self._p.resetSimulation()

        self._p.setPhysicsEngineParameter(fixedTimeStep=self.timeStep * 1,
                                          numSolverIterations=self.numSolverIterations,
                                          numSubSteps=1)
        self._p.setGravity(0, 0, self.gravity)

    def createWalls(self):
        "create ground"
        self.ground = self._p.loadURDF(self._urdf_root + "plane.urdf")
        orange, yellow, green, blue, purple = list(np.array([247, 159, 45]) / 255.), list(
            np.array([252, 207, 3, 255]) / 255.), list(
            np.array([3, 252, 34, 255]) / 255.), list(np.array([3, 136, 252, 255]) / 255.), list(
            np.array([157, 3, 252, 255]) / 255.)
        # Add walls
        # Path to the urdf file
        if self.class_name == 'TurtlebotMazeEnv':
            wallMaze1_urdf = os.path.join(self.script_path, "urdf/wallMaze1.urdf")
            unit = 1.50
            wall_left = self._p.loadURDF(wallMaze1_urdf, [unit * 4 / 2, 0, 0], useFixedBase=True,
                                         flags=pybullet.URDF_USE_SELF_COLLISION)
            self._p.changeVisualShape(wall_left, -1, rgbaColor=yellow)

            wall_right = self._p.loadURDF(wallMaze1_urdf, [unit * 4 / 2, unit * 3, 0], useFixedBase=True,
                                          flags=pybullet.URDF_USE_SELF_COLLISION)
            self._p.changeVisualShape(wall_right, -1, rgbaColor=green)

            wallMaze2_urdf = os.path.join(self.script_path, "urdf/wallMaze2.urdf")

            # top wall
            wall_top = self._p.loadURDF(wallMaze2_urdf, [0., unit * 3 / 2, 0],
                                        self._p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                        flags=pybullet.URDF_USE_SELF_COLLISION)
            self._p.changeVisualShape(wall_top, -1, rgbaColor=purple)

            wall_inLeft = self._p.loadURDF(wallMaze2_urdf, [unit + unit * 3 / 2, unit, 0], useFixedBase=True,
                                           flags=pybullet.URDF_USE_SELF_COLLISION)
            self._p.changeVisualShape(wall_inLeft, -1, rgbaColor=blue)

            wall_inRight = self._p.loadURDF(wallMaze2_urdf, [unit + unit * 3 / 2, 2 * unit, 0], useFixedBase=True,
                                            flags=pybullet.URDF_USE_SELF_COLLISION)
            self._p.changeVisualShape(wall_inRight, -1, rgbaColor=blue)

            wallMaze3_urdf = os.path.join(self.script_path, "urdf/wallMaze3.urdf")

            wall_bottom = self._p.loadURDF(wallMaze3_urdf, [unit, unit * 3 / 2, 0],
                                           self._p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                           flags=pybullet.URDF_USE_SELF_COLLISION)
            self._p.changeVisualShape(wall_bottom, -1, rgbaColor=blue)

            wall_bottomLeft = self._p.loadURDF(wallMaze3_urdf, [unit * 4, unit / 2, 0],
                                               self._p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                               flags=pybullet.URDF_USE_SELF_COLLISION)
            self._p.changeVisualShape(wall_bottomLeft, -1, rgbaColor=green)

            wall_bottomRight = self._p.loadURDF(wallMaze3_urdf, [unit * 4, unit * 2 + unit / 2, 0],
                                                self._p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                                flags=pybullet.URDF_USE_SELF_COLLISION)
            self._p.changeVisualShape(wall_bottomRight, -1, rgbaColor=yellow)

            self.walls = [wall_left, wall_bottom, wall_right, wall_top, wall_inLeft, wall_inRight, wall_bottomLeft,
                          wall_bottomRight]
        else:
            wall_urdf = wallMaze2_urdf = os.path.join(self.script_path,
                                                      "urdf/wall_little.urdf")

            wall_left = self._p.loadURDF(wall_urdf, [self._max_x / 2, 0, 0], useFixedBase=True,
                                         flags=pybullet.URDF_USE_SELF_COLLISION)
            # Change color
            self._p.changeVisualShape(wall_left, -1, rgbaColor=yellow)

            # getQuaternionFromEuler -> define orientation
            wall_bottom = self._p.loadURDF(wall_urdf, [self._max_x, self._max_y / 2, 0],
                                           self._p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                           flags=pybullet.URDF_USE_SELF_COLLISION)
            self._p.changeVisualShape(wall_bottom, -1, rgbaColor=purple)

            wall_right = self._p.loadURDF(wall_urdf, [self._max_x / 2, self._max_y, 0], useFixedBase=True,
                                          flags=pybullet.URDF_USE_SELF_COLLISION)
            self._p.changeVisualShape(wall_right, -1, rgbaColor=green)

            wall_top = self._p.loadURDF(wall_urdf, [self._min_x, self._max_y / 2, 0],
                                        self._p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                        flags=pybullet.URDF_USE_SELF_COLLISION)
            self._p.changeVisualShape(wall_top, -1, rgbaColor=blue)

            self.walls = [wall_left, wall_bottom, wall_right, wall_top]

    def resetRobot(self):
        self.forward = 0
        self.turn = 0
        self.leftWheelVelocity = 0
        self.rightWheelVelocity = 0

        if 'plate' in self.noise_type:
            self.noise_adder.reset(self._p)

        if self.with_target:
            "Initialize target position"
            if self.debug:
                self.target_pos = np.array((1.5 * self.wallLimit, 3 * 1.50 - 1.5 * self.wallLimit, 0))
            elif self.random_target:
                self.target_pos = self.init_goal()
            elif self.target_pos is None:
                self.target_pos = self.init_goal()

            if self.target_uid is None:
                if self.display_target:
                    self.target_uid = self._p.loadURDF(os.path.join(self.script_path, "urdf/cylinder.urdf"), self.target_pos,
                                                       useFixedBase=True)
                else:
                    self.target_uid = self._p.loadURDF(os.path.join(self.script_path, "urdf/cylinder_invisible.urdf"),
                                                       self.target_pos,
                                                       useFixedBase=True)
            self._p.resetBasePositionAndOrientation(self.target_uid, self.target_pos, [0, 0, 0, 1])

        "Add mobile robot and target"
        too_close = True
        if self.randomExplor:
            "Init the robot randomly"
            while too_close:
                turtleStartPos = self.init_goal()
                turtleStartPos[-1] = initZ
                if self.with_target:
                    too_close = self.goal_distance(turtleStartPos[:-1], self.target) < self.target_radius * 3
                else:
                    too_close = False
        else:
            if self.class_name == 'TurtlebotMazeEnv':
                "place the robot in the bottom wall_right"
                turtleStartPos = np.array((1.50 * 3 + 1.50 / 2, 1.50 * 2 + 1.50 / 2, initZ))
            else:
                turtleStartPos = np.array(
                    [self._max_x / 2 - self._max_x / 10, self._max_y / 2 + self._max_y / 10, initZ])
        if self.debug:
            self.theta = 25 / 180.0 * np.pi
        else:
            self.theta = self.np_random.uniform(low=0, high=2 * np.pi)

        self.turtleStartOrientation = self._p.getQuaternionFromEuler([0, 0, self.theta])

        "Create robot"
        if self.turtle is None:
            black, grey1, grey2, white, red = [0., 0., 0., 1.], [0.33, 0.33, 0.33, 1], [0.66, 0.66, 0.66, 1], [1., 1.,
                                              1.,1], [1.,.3,.3,1]
            self.turtle = self._p.loadURDF(os.path.join(self.script_path, "urdf/turtlebot.urdf"), turtleStartPos,
                                           self.turtleStartOrientation,
                                           flags=pybullet.URDF_USE_SELF_COLLISION)
            self._p.changeVisualShape(self.turtle, 26, rgbaColor=black)
            self._p.changeVisualShape(self.turtle, 29, rgbaColor=red)
            for i in range(-1, 26):
                self._p.changeVisualShape(self.turtle, i, rgbaColor=white)
        else:
            self._p.resetBasePositionAndOrientation(self.turtle, turtleStartPos, self.turtleStartOrientation)

    def reset(self):
        self._reset()

        self.info = {}
        self.terminated = False
        self.has_bumped = False

        if self.walls is None:
            self.createWalls()

        self.resetRobot()

        self.envStepCounter = 0
        self._observation = self.getExtendedObservation()
        "Init positions"
        self.robot_pos = np.array(self._p.getBasePositionAndOrientation(self.turtle)[0])

        return np.hstack(self._observation)

    def resetPosition(self, pos, theta=None):
        if theta is None:
            self.theta = 0
            orientation = [0, 0, 0, 1]
        else:
            self.theta = theta
            orientation = self._p.getQuaternionFromEuler([0, 0, theta])
        self._p.resetBasePositionAndOrientation(self.turtle, pos, orientation)
        self.robot_pos = np.array(self._p.getBasePositionAndOrientation(self.turtle)[0])

    def bump_detection(self, a_, bump=True):
        has_bumped = False
        assert np.abs(a_[0]) <= 1, 'action above bounds'
        assert np.abs(a_[1]) <= 1, 'action above bounds'

        for i in range(self._actionRepeat):
            # copy robot_pos to avoid changing real robot position!
            robot_pos = self.robot_pos.copy()
            robot_pos[:2] += a_ * self.posMultiplier
            # Handle collisions
            if bump:
                has_bumped_detected = self.detect_collision(robot_pos)
        if bump:
            return has_bumped_detected
        else:
            return robot_pos[:2]

    def endDetector(self, debug=False):
        endDetection = False
        if debug:
            if self.robot_pos[0] < 4.8:  # abscissa/ ordinate
                endDetection = True
        else:
            if self.robot_pos[0] > 5.23 and self.robot_pos[1] < 1.04987811:  # abscissa/ ordinate
                endDetection = True
        return endDetection

    def detect_collision(self, robot_pos):
        margin = self.wallLimit  # 0.155 * self._max_x
        has_bumped = False
        if self.class_name == 'TurtlebotMazeEnv':
            max_x, max_y = 1.50 * 4, 1.50 * 3
            if robot_pos[0] > 1.50 - margin:
                if robot_pos[1] < 2 * 1.50 + margin and robot_pos[1] > 1.50 - margin:
                    has_bumped = True
        else:
            max_x, max_y = self._max_x, self._max_y

        if not has_bumped:
            for i, limit in enumerate([max_x, max_y]):
                # If it has bumped against a wall, stay at the previous position
                if robot_pos[i] < margin or robot_pos[i] > limit - margin:
                    has_bumped = True
                    break
        return has_bumped

    def step(self, action_):

        if self.wallDistractor:
            c = list(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255]) / 255.)
            self._p.changeVisualShape(self.walls[-1], -1, rgbaColor=c)
        self.has_bumped = False

        assert np.abs(action_[0]) <= 1, 'action above bounds'
        assert np.abs(action_[1]) <= 1, 'action above bounds'
        if False in (action_ == np.zeros((2))):
            self.theta = np.arctan2(action_[1], action_[0])
        else:
            self.theta = self.theta
        for i in range(self._actionRepeat):
            self.envStepCounter += 1

            self.previous_pos = self.robot_pos.copy()
            self.robot_pos[:2] += action_ * self.posMultiplier
            # Handle collisions
            self.has_bumped = self.detect_collision(self.robot_pos)
            if self.has_bumped:
                self.robot_pos = self.previous_pos

            orientation = self._p.getQuaternionFromEuler([0, 0, self.theta])
            self._p.resetBasePositionAndOrientation(self.turtle, self.robot_pos, orientation)

            if self.renders:
                time.sleep(self.timeStep)

        self._observation = self.getExtendedObservation()
        reward = self._reward() - int(self.has_bumped)
        done = self._termination()
        self.info['has_bumped'] = self.has_bumped
        return np.hstack(self._observation), reward, done, self.info

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

    def get_info(self):
        self.info = {}
        observation = []
        # Position
        observation.extend(self.object)
        # Orientation
        _, carOrn = self._p.getBasePositionAndOrientation(self.turtle)
        EulZ = self._p.getEulerFromQuaternion(carOrn)[2]
        EulZ = EulZ % (2 * np.pi)
        observation.extend([EulZ])
        keys = ['robotX', 'robotY', 'orn']

        self.info['labels'] = OrderedDict((keys[id], observation[id]) for id in range(len(observation)))
        self.info['angles'] = EulZ
        if self.with_target:
            self.info['target'] = self.target
        return self.info

    def measureRobot(self):
        return self.getExtendedObservation()

    def getExtendedObservation(self):
        observation = np.hstack((self.object,
                                 self._p.getEulerFromQuaternion(self._p.getBasePositionAndOrientation(self.turtle)[1])[
                                     2] % (2 * np.pi)))
        if not self.GoalWrapper and self.random_target:
            np.array((observation.extend(list(self.target))))
        return observation

    @property
    def target(self):
        # Return only the [x, y] coordinates of the target
        self._target = np.array(self._p.getBasePositionAndOrientation(self.target_uid)[0][:2])
        return self._target

    @property
    def object(self):
        # Return only the [x, y] coordinates of the robot
        self._object = np.array(self._p.getBasePositionAndOrientation(self.turtle)[0][:2])
        return self._object

    @property
    def target_radius(self):
        return 0.20

