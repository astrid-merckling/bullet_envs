import numpy as np


NOISES = ["none", "noisyObs", "random_noise", "random_image", "random_cutout"]

env_with_goals = ["ReacherBulletEnv-v0", "TurtlebotEnv-v0", "TurtlebotMazeEnv-v0"]
env_with_fpv = ["TurtlebotEnv-v0", "TurtlebotMazeEnv-v0"]
env_with_distractor = ["ReacherBulletEnv-v0"]

PY_MUJOCO = [
    "InvertedPendulumSwingupBulletEnv-v0",
    "InvertedDoublePendulumBulletEnv-v0",
    "ReacherBulletEnv-v0",
    "PybulletEnv-v0",
    "AntBulletEnv-v0",
    "HalfCheetahBulletEnv-v0",
    "HopperBulletEnv-v0",
    "Walker2DBulletEnv-v0",
]
PYBULLET_ENV = ["TurtlebotEnv-v0", "TurtlebotMazeEnv-v0"] + PY_MUJOCO
ENV_ALIVE = [
    "AntBulletEnv-v0",
    "HalfCheetahBulletEnv-v0",
    "HopperBulletEnv-v0",
    "Walker2DBulletEnv-v0",
    "InvertedDoublePendulumBulletEnv-v0",
]


def seeding_np_random(seed):
    rng = np.random.RandomState(seed=seed)
    return rng, seed


class AddNoise(object):
    def __init__(self, config):
        self.noise_type = config["noise_type"]
        self.image_size = config["image_size"]
        if "random_state" not in config:
            try:
                self.random_state = np.random.RandomState(seed=config["_seed"])
            except:
                self.random_state = np.random.RandomState(seed=config["seed"])
        else:
            self.random_state = config["random_state"]

        if "image" in self.noise_type:
            self.im_counts = 0
            self.data_num = 0
            self.image_path = (
                lambda x: "path2cifar/cifar-10-batches-py/data_batch_%s" % (x % 5 + 1)
            )
            self.images_for_noise = load_cifar10(self.image_path(self.data_num))
            self.data_num += 1

    def __call__(self, p_=None, im_shapes=None, observation=None, action=None):
        normalized = False
        if np.max(observation) <= 1:
            normalized = True
            observation = observation * 255
        if self.noise_type == "noisyObs":
            out = observation + self.random_state.normal(
                0, 30, (observation.shape)
            ).astype(observation.dtype)
            out = np.minimum(np.maximum(out, 0), 255)

        elif "cutout" in self.noise_type:
            out = cutout(observation, n_holes=1, length=int(self.image_size / 2.0))
        else:
            print("TODO")
            exit()
            if "random_image" == self.noise_type:
                self.im_counts += 1
                self.load_image()
            elif "random_noise" == self.noise_type:
                self.current_noise = self.random_state.randint(
                    0,
                    255,
                    (
                        int(observation.shape[0] / 2),
                        int(observation.shape[1] / 2),
                        observation.shape[2],
                    ),
                )
            out = observation
            out[
                int(observation.shape[0] / 2) :, int(observation.shape[1] / 2) :, :
            ] = self.current_noise

        if normalized:
            out = out / 255.0
        return out

    def load_image(self):
        self.current_noise = self.images_for_noise[
            self.random_state.randint(0, len(self.images_for_noise))
        ]
        if (
            self.im_counts % 10000 == 0
        ):  # each of the batch files contains 10000 32*32 color images
            self.images_for_noise = load_cifar10(self.image_path(self.data_num))
            self.data_num += 1
