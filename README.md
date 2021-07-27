

# bullet_envs

It is a library that provides environments similar to those provided by MuJoCo which are fully compatible with [OpenAI GYM](https://arxiv.org/abs/1606.01540), with the difference of being open-source.

It is based on the original environments provided in [pybullet_envs](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs).


Only the following environments are fully supported:
* `TurtlebotMazeEnv-v0`,
* `ReacherBulletEnv-v0`,
* `HalfCheetahBulletEnv-v0`,
* `InvertedPendulumSwingupBulletEnv-v0`}

To add a new environment, simply add a new entry to the [__init__.py](./bullet_ens/__init__.py) file.

# Contributions

* All environments have a camera rendering in `def render` which is wrapped into an OpenAI gym wrapper.

* `TurtlebotMazeEnv-v0` is proposed here as a new environment, built from the original `Turtebot` implemented in [`pybullet_robots`](https://github.com/erwincoumans/pybullet_robots).

* `ReacherBulletEnv-v0` has a new version with a randomly moving ball as a distractor which corresponds to the file `reacher_distractor.xml`


# Installation


Clone this repo and have its path added to your `PYTHONPATH` environment variable:
```bash
cd <installation_path_of_your_choice>
git clone https://github.com/astrid-merckling/bullet_envs.git
cd bullet_envs
export PYTHONPATH=$(pwd):${PYTHONPATH}
```
