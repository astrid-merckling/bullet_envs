
# Watch a Demo

[[Watch a presentation]](https://youtu.be/yejK8RmTfwE)
<!-- 
https://user-images.githubusercontent.com/62666911/127876105-f851cf18-5793-4cff-ad18-fb53634d36dc.mp4 -->



# bullet_envs

This library extends some original [PyBullet](http://pybullet.org) environments provided in [pybullet_envs](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs).

[PyBullet](http://pybullet.org) environments are similar to those provided by [MuJoCo](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.6848&rep=rep1&type=pdf) which are fully compatible with [OpenAI Gym](https://arxiv.org/abs/1606.01540), with the difference of being open-source.


Only the following environments are fully supported:
* `TurtlebotMazeEnv-v0`
* `ReacherBulletEnv-v0`
* `HalfCheetahBulletEnv-v0`
* `InvertedPendulumSwingupBulletEnv-v0`

# Contributions

* All environments have a camera rendering in `def render` which is wrapped into an OpenAI gym wrapper.

* `TurtlebotMazeEnv-v0` is proposed here as a new environment, built from the original `Turtebot` implemented in [pybullet_robots](https://github.com/erwincoumans/pybullet_robots). It includes a version  where one of the walls has a randomly sampled color at each time step.
The observation space corresponds to a first-person perspective camera.

* `ReacherBulletEnv-v0` has a new version with a randomly moving ball as a distractor which corresponds to the file `reacher_distractor.xml`


# Installation


Clone this repo and have its path added to your `PYTHONPATH` environment variable:
```bash
cd <installation_path_of_your_choice>
git clone https://github.com/astrid-merckling/bullet_envs.git
cd bullet_envs
export PYTHONPATH=$(pwd):${PYTHONPATH}
```


You can install the dependencies as:
```bash
pip install gym==0.17.2
pip install pybullet==2.6.4
pip install opencv-python==4.1.2.30
```


# Usage

Example to run and visualize `ReacherBulletEnv-v0` with a randomly moving ball (`distractor=True`), where the observation space is chosen to be the camera:
```python
import gym

"register bullet_envs in gym"
import bullet_envs.__init__

env_name = 'ReacherBulletEnv-v0'
actionRepeat = 1
maxSteps = 50
"OpenAI Gym env creation"
env = gym.make('PybulletEnv-v0', env_name=env_name, renders=True, distractor=True, actionRepeat=actionRepeat,
               maxSteps=maxSteps * actionRepeat, image_size=64, display_target=True)

"running env on 5 episodes"
num_ep = 5
for episode in range(num_ep):
    obs = env.reset()
    done = False
    while not done:
        # follow a random policy
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        "get the image observation from the camera"
        obs = env.render()
```

Example to run and visualize `TurtlebotMazeEnv-v0` with a randomly sampled wall color (`wallDistractor=True`), where the observation space is chosen to be the first-person camera:
```python
import gym

"register bullet_envs in gym"
import bullet_envs.__init__

env_name = 'TurtlebotMazeEnv-v0'
actionRepeat = 1
maxSteps = 100
"OpenAI Gym env creation"
env = gym.make(env_name, renders=True, wallDistractor=True, maxSteps=maxSteps, image_size=64, display_target=True)

"running env on 2 episodes"
num_ep = 2
for episode in range(num_ep):
    obs = env.reset()
    done = False
    while not done:
        # follow a random policy
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        "get the image observation from the camera"
        obs = env.render()
```

See [OpenAI Gym](https://github.com/openai/gym) for more details on the Python `env` class.



<!-- InvertedPendulum and HalfCheetah belong to the MuJoCo torque-controlled benchmark implemented in PyBullet (http://pybullet.org).

We implemented the new TurtleBot Maze environment in PyBullet, where the observation space corresponds to a first-person perspective camera. -->
