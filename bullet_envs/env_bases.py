import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet
import os
import cv2

from bullet_envs.utils import AddNoise

from pybullet_utils import bullet_client

from pkg_resources import parse_version

try:
    if os.environ["PYBULLET_EGL"]:
        import pkgutil
except:
    pass

STATE_W = 96
STATE_H = 96


class MJCFBaseBulletEnv(gym.Env):
    """
    Base class for Bullet physics simulation loading MJCF (MuJoCo .xml) environments in a Scene.
    These environments create single-player scenes and behave like normal Gym environments, if
    you don't use multiplayer.
    """

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

    def __init__(self, robot, renders=False, doneAlive=True, actionRepeat=1,seed=None):
        self.actionRepeat = actionRepeat
        self.doneAlive = doneAlive
        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.camera = Camera()
        self.renders = renders
        self.robot = robot
        self.seedValue = seed
        self.seed(seed)
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = 0 #old:-30 more simple to predict camera with zero pitch
        # self._render_width = 320
        # self._render_height = 240

        self.action_space = robot.action_space
        self.observation_space = robot.observation_space
        self._cam_pos = [0,0.5,2.]
        if self.__class__.__name__ == 'InvertedPendulumSwingupBulletEnv':
            self._cam_pos=[0,0.5,1]#[0,1.8,.0]
        elif self.__class__.__name__ == 'InvertedDoublePendulumBulletEnv':
            self._cam_pos = [0, 0.2, 1]
        elif self.__class__.__name__ in ['AntBulletEnv']:
            "TODO later"
            self._cam_pitch = -60
            self._cam_yaw = 0
            self._cam_dist = 2.5
        elif self.__class__.__name__ == 'ReacherBulletEnv':
            self._cam_pos = [0., -0.1, -0.]
            self._cam_dist = 0.9
            self._cam_yaw = 2
            self._cam_pitch = -25

    def configureCamera(self, config,image_size=64, noise_type='none', color=True):
        self.image_size = image_size
        self.noise_type = noise_type
        self.color = color
        if self.noise_type != 'none':
            self.noise_adder = AddNoise(config)


    # def configure(self, args):
    #     self.robot.args = args


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]


    def reset(self):
        if (self.physicsClientId < 0):
            self.ownsPhysicsClient = True

            if self.renders:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
                self._p.resetDebugVisualizerCamera(cameraTargetPosition=self._cam_pos,
                                                   cameraDistance=self._cam_dist,
                                                   cameraYaw=self._cam_yaw,
                                                   cameraPitch=self._cam_pitch)
            else:
                self._p = bullet_client.BulletClient()
            self._p.resetSimulation()
            # optionally enable EGL for faster headless rendering
            try:
                if os.environ["PYBULLET_EGL"]:
                    con_mode = self._p.getConnectionInfo()['connectionMethod']
                    if con_mode == self._p.DIRECT:
                        egl = pkgutil.get_loader('eglRenderer')
                        if (egl):
                            self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                        else:
                            self._p.loadPlugin("eglRendererPlugin")
            except:
                pass
            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        self.robot.scene = self.scene

        self.frame = 0
        self.done = 0
        dump = 0
        s = self.robot.reset(self._p)
        "calc_state() always to be executed before calc_potential()"
        self.robot.calc_state()
        self.potential = self.robot.calc_potential()
        self.base_pos = [0, 0, 0]
        return s


    def render(self, mode='rgb_array', image_size=None, color=None, close=False,camera_id=0,downscaling=True):

        if downscaling:
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            VP_W = image_size
            VP_H = image_size

        image_size = self.image_size if image_size is None else image_size
        color = self.color if color is None else color
        im_shapes = [image_size, image_size, 3] if color else [image_size, image_size, 1]

        cam_pos = self._cam_pos.copy()
        cam_dist = self._cam_dist
        cam_yaw = self._cam_yaw
        cam_pitch = self._cam_pitch
        if camera_id >= 0:
            if self.__class__.__name__ == 'ReacherBulletEnv':
                cam_pos = [0, 0., 0.]
                cam_pitch = -90
                cam_yaw = 0
                cam_dist = 0.5
            if self.__class__.__name__ in ['InvertedPendulumSwingupBulletEnv', 'InvertedDoublePendulumBulletEnv']:
                if self.__class__.__name__ == 'InvertedPendulumSwingupBulletEnv':
                    cam_pos = [0, 1.5, 1]
                elif self.__class__.__name__ == 'InvertedDoublePendulumBulletEnv':
                    cam_pos = [0, 1, 1]
                cam_pos[0], _ = self.robot.slider.current_position()
        if (hasattr(self, 'robot')):
            if (hasattr(self.robot, 'body_xyz')):
                cam_pos = self.robot.robot_body.pose().xyz()
                if self.__class__.__name__ == 'HalfCheetahBulletEnv':
                    cam_pos[1] += 1.
                    cam_pos[2] = 1.
                elif self.__class__.__name__ in ['HopperBulletEnv', 'Walker2DBulletEnv']:
                    cam_pos[1] += 1.
                    cam_pos[2] = 1
                elif self.__class__.__name__ in ['AntBulletEnv']:
                    cam_pos[1] += 0.
                    cam_pos[2] = 0.5


        if mode == "human":
            if self.__class__.__name__ in ['HalfCheetahBulletEnv','HopperBulletEnv', 'Walker2DBulletEnv','AntBulletEnv']:
                self._p.resetDebugVisualizerCamera(cameraTargetPosition=cam_pos,
                                                   cameraDistance=cam_dist,
                                                   cameraYaw=cam_yaw,
                                                   cameraPitch=cam_pitch,)
            return np.array([])
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=cam_pos,
                                                                distance=cam_dist,
                                                                yaw=cam_yaw,
                                                                pitch=cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(VP_W) /VP_H,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=VP_W,
                                                  height=VP_H,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px)[:, :, :3].astype(np.float32)

        if not color:
            rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
            rgb_array = np.expand_dims(rgb_array, -1)
        if self.noise_type in ['noisyObs', 'random_noise', 'random_image', 'random_cutout']:
            rgb_array = self.noise_adder(observation=rgb_array)

        if image_size != VP_W:
            "Resize to a square image with rgb_array values between 0 and 255"
            rgb_array = cv2.resize(rgb_array.astype(np.uint8), dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)

        return rgb_array


    def close(self):
        if (self.ownsPhysicsClient):
            if (self.physicsClientId >= 0):
                self._p.disconnect()
        self.physicsClientId = -1


    def HUD(self, state, a, done):
        pass

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed


class Camera:

    def __init__(self):
        pass

    def move_and_look_at(self, i, j, k, x, y, z):
        lookat = [x, y, z]
        distance = 10
        yaw = 10
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, lookat)
