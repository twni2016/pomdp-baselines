import gym
from gym import spaces, logger
import math
import numpy as np
from gym.utils import seeding
from copy import deepcopy
import warnings
import os

try:
    import roboschool
except:
    pass


class ContinuousCartPoleEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ]
        )

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        if not self.action_space.contains(action):
            warnings.warn("%r (%s) invalid" % (action, type(action)))

        action = action.clip(self.action_space.low, self.action_space.high)
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    """
                            You are calling 'step()' even though this environment has already returned
                            done = True. You should always call 'reset()' once you receive 'done = True'
                            Any further steps are undefined behavior.
                            """
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == "rgb_array"))

    def close(self):
        if self.viewer:
            self.viewer.close()


class PendulumP:
    """
    Partially observed Pendulum:
    Only the current angle is observed
    """

    def __init__(self):

        self.env = gym.make("Pendulum-v0")
        self.red_position = 0
        self.green_position = 0
        self.blue_position = 0

        self.action_space = spaces.Box(low=-2, high=2, shape=(1,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))

    def reset(self):
        obs = self.env.reset()
        self.old_position = obs[0:2]
        return obs[0:2]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        self.new_position = obs[0:2]
        self.old_position = deepcopy(self.new_position)
        return obs[0:2], r, done, {}

    def render(self, mode="human"):
        self.env.render()


class TaskT(gym.Env):
    metadata = {
        "name": "TaskT",
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50,
    }
    spec = {"id": "TaskT"}

    def __init__(
        self,
        sections=1,
        seq="RGB",
        final_reward=False,
        reward_obs=True,
        R=4,
        saving=False,
        log_dir="./TaskT_log/",
        reward_scales=[20.0, 50.0, 100.0],
    ):
        """
        Sequential target reaching task.
        :param sections: how many targets to reach to finish the task
        :param seq: any combination of 'R', 'G', 'B' to indicated the required sequence of target-reaching.
        :param final_reward: if True, only final target provides reward, otherwise all targets provide reward.
        :param reward_obs: whether reward is one element of observation
        :param R: difficulty (distance between targets)
        :param saving: whether to save steps/rewards into txt file
        :param log_dir: directory to save steps/rewards
        """
        self.sections = sections
        self.saving = saving
        self.log_dir = log_dir
        self.final_reward = final_reward
        self.reward_obs = reward_obs
        self.sequence = seq
        self.R = R
        self.reward = 0.0
        self.reward_signal = 0.0
        self.dim_position = 2
        self.dim_action = 2
        self.speed = 0.8
        self.radius = 0.5
        self.max_steps = 128
        self.steps = 0

        self.reward1 = reward_scales[0]
        self.reward2 = reward_scales[1]
        self.reward3 = reward_scales[2]

        self.init_position = np.array([7.5, 7.5], dtype=np.float32)
        self.init_position[0] += np.float32(15 * (np.random.rand() - 0.5))
        self.init_position[1] += np.float32(15 * (np.random.rand() - 0.5))
        self.old_position = self.init_position
        self.new_position = self.init_position
        self.orientation = 2 * np.pi * np.random.rand()

        self.init_state = 0
        self.size = 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        if reward_obs:
            self.observation_space = spaces.Box(low=-1.0, high=5.0, shape=(12,))
        else:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(11,))

        self.reward_range = (-np.Inf, np.Inf)

        self._seed()

        if self.saving:
            if os.path.exists(log_dir):
                warnings.warn("{} exists (possibly so do data).".format(log_dir))
            else:
                os.makedirs(log_dir)

            path = self.log_dir + "TaskT" + ".txt"
            self.file_pointer = open(path, "w+")

        self.red_position = np.float32(
            R * (np.random.rand(self.dim_position) - 0.5)
        ) + np.array([7.5, 7.5], dtype=np.float32)
        while True:
            self.green_position = np.float32(
                R * (np.random.rand(self.dim_position) - 0.5)
            ) + np.array([7.5, 7.5], dtype=np.float32)
            if (np.sum((self.red_position - self.green_position) ** 2)) > 2:
                break
        while True:
            self.blue_position = np.float32(
                R * (np.random.rand(self.dim_position) - 0.5)
            ) + np.array([7.5, 7.5], dtype=np.float32)
            if (np.sum((self.blue_position - self.green_position) ** 2)) > 2 and (
                np.sum((self.blue_position - self.red_position) ** 2)
            ) > 2:
                break

        self.first_experience = 0
        self.second_experience = 0
        self.third_experience = 0

        if self.sequence[0] == "R":
            self.first_position = self.red_position
        elif self.sequence[0] == "G":
            self.first_position = self.green_position
        elif self.sequence[0] == "B":
            self.first_position = self.blue_position

        if self.sections >= 2:
            if self.sequence[1] == "R":
                self.second_position = self.red_position
            elif self.sequence[1] == "G":
                self.second_position = self.green_position
            elif self.sequence[1] == "B":
                self.second_position = self.blue_position

        if self.sections >= 3:
            if self.sequence[2] == "R":
                self.third_position = self.red_position
            elif self.sequence[2] == "G":
                self.third_position = self.green_position
            elif self.sequence[2] == "B":
                self.third_position = self.blue_position

        self.done = 0

        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.reward = 0.0
        self.steps = 0
        self.reward_signal = 0.0
        self.init_position = np.array([7.5, 7.5], dtype=np.float32)
        self.init_position[0] += np.float32(15 * (np.random.rand() - 0.5))
        self.init_position[1] += np.float32(15 * (np.random.rand() - 0.5))
        self.old_position = self.init_position
        self.new_position = self.init_position
        self.orientation = 2 * np.pi * np.random.rand()

        R = self.R
        self.red_position = np.float32(
            R * (np.random.rand(self.dim_position) - 0.5)
        ) + np.array([7.5, 7.5], dtype=np.float32)
        while True:
            self.green_position = np.float32(
                R * (np.random.rand(self.dim_position) - 0.5)
            ) + np.array([7.5, 7.5], dtype=np.float32)
            if (np.sum((self.red_position - self.green_position) ** 2)) > 2:
                break
        while True:
            self.blue_position = np.float32(
                R * (np.random.rand(self.dim_position) - 0.5)
            ) + np.array([7.5, 7.5], dtype=np.float32)
            if (np.sum((self.blue_position - self.green_position) ** 2)) > 2 and (
                np.sum((self.blue_position - self.red_position) ** 2)
            ) > 2:
                break

        self.first_experience = 0
        self.second_experience = 0
        self.third_experience = 0

        if self.sequence[0] == "R":
            self.first_position = self.red_position
        elif self.sequence[0] == "G":
            self.first_position = self.green_position
        elif self.sequence[0] == "B":
            self.first_position = self.blue_position

        if self.sections >= 2:
            if self.sequence[1] == "R":
                self.second_position = self.red_position
            elif self.sequence[1] == "G":
                self.second_position = self.green_position
            elif self.sequence[1] == "B":
                self.second_position = self.blue_position

        if self.sections >= 3:
            if self.sequence[2] == "R":
                self.third_position = self.red_position
            elif self.sequence[2] == "G":
                self.third_position = self.green_position
            elif self.sequence[2] == "B":
                self.third_position = self.blue_position

        self.done = 0
        return self.get_obs()

    def get_obs(self):

        lambd = 3.0

        position = self.new_position
        theta = self.orientation

        red_dis = np.sqrt(np.sum((position - self.red_position) ** 2))
        green_dis = np.sqrt(np.sum((position - self.green_position) ** 2))
        blue_dis = np.sqrt(np.sum((position - self.blue_position) ** 2))

        if 0 <= theta and theta < np.pi / 2:
            dw1 = min(
                (15 - position[1]) / abs(np.sin(theta)),
                (15 - position[0]) / abs(np.cos(theta)),
            )
            dw2 = min(
                (position[1] - 0) / abs(np.sin(theta)),
                (position[0] - 0) / abs(np.cos(theta)),
            )
        elif np.pi / 2 <= theta and theta < np.pi:
            dw1 = min(
                (15 - position[1]) / abs(np.sin(theta)),
                (position[0] - 0) / abs(np.cos(theta)),
            )
            dw2 = min(
                (position[1] - 0) / abs(np.sin(theta)),
                (15 - position[0]) / abs(np.cos(theta)),
            )
        elif np.pi <= theta and theta < 3 * np.pi / 2:
            dw1 = min(
                (position[1] - 0) / abs(np.sin(theta)),
                (position[0] - 0) / abs(np.cos(theta)),
            )
            dw2 = min(
                (15 - position[1]) / abs(np.sin(theta)),
                (15 - position[0]) / abs(np.cos(theta)),
            )
        else:
            dw1 = min(
                (position[1] - 0) / abs(np.sin(theta)),
                (15 - position[0]) / abs(np.cos(theta)),
            )
            dw2 = min(
                (15 - position[1]) / abs(np.sin(theta)),
                (position[0] - 0) / abs(np.cos(theta)),
            )

        tr = (
            np.arctan2(
                self.red_position[1] - position[1], self.red_position[0] - position[0]
            )
            - theta
        )
        tg = (
            np.arctan2(
                self.green_position[1] - position[1],
                self.green_position[0] - position[0],
            )
            - theta
        )
        tb = (
            np.arctan2(
                self.blue_position[1] - position[1], self.blue_position[0] - position[0]
            )
            - theta
        )

        if self.reward_obs:
            obs = np.array(
                [
                    np.exp(-red_dis / lambd),
                    np.exp(-green_dis / lambd),
                    np.exp(-blue_dis / lambd),
                    np.exp(-dw1 / lambd),
                    np.exp(-dw2 / lambd),
                    np.sin(tr),
                    np.cos(tr),
                    np.sin(tg),
                    np.cos(tg),
                    np.sin(tb),
                    np.cos(tb),
                    self.reward_signal,
                ]
            )
        else:
            obs = np.array(
                [
                    np.exp(-red_dis / lambd),
                    np.exp(-green_dis / lambd),
                    np.exp(-blue_dis / lambd),
                    np.exp(-dw1 / lambd),
                    np.exp(-dw2 / lambd),
                    np.sin(tr),
                    np.cos(tr),
                    np.sin(tg),
                    np.cos(tg),
                    np.sin(tb),
                    np.cos(tb),
                ]
            )
        return obs

    def get_init_position(self):
        return self.init_position

    def reward_fun(self):

        position = self.old_position
        new_position = self.new_position

        if (
            new_position[0] > 15
            or new_position[0] < 0
            or new_position[1] > 15
            or new_position[1] < 0
        ):
            r = -0.1
        # self.done = 1
        else:
            if not self.first_experience:
                target_position = self.first_position
                dis2 = np.sum((new_position - target_position) ** 2)
                if dis2 < 0.16:
                    self.first_experience = 1
                    if self.sections == 1:
                        self.done = 1
                    r = (
                        self.reward1 / (1 + np.sqrt(dis2))
                        if (self.sections == 1 or (not self.final_reward))
                        else 0.0
                    )
                else:
                    r = 0.0
            elif self.sections >= 2 and (not self.second_experience):
                target_position = self.second_position
                dis2 = np.sum((new_position - target_position) ** 2)
                if dis2 < 0.16:
                    self.second_experience = 1
                    if self.sections == 2:
                        self.done = 1
                    r = (
                        self.reward2 / (1 + np.sqrt(dis2))
                        if (self.sections == 2 or (not self.final_reward))
                        else 0.0
                    )
                else:
                    r = 0.0
            elif self.sections >= 3:
                target_position = self.third_position
                dis2 = np.sum((new_position - target_position) ** 2)
                if dis2 < 0.16:
                    self.third_experience = 1
                    if self.sections == 3:
                        self.done = 1
                    r = (
                        self.reward3 / (1 + np.sqrt(dis2))
                        if self.sections == 3
                        else 0.0
                    )
                else:
                    r = 0.0
            else:
                r = 0.0
        return r

    def step(self, action, saving=None):

        if self.done:
            warnings.warn("Task already done!! Not good to continue!")

        if saving is None:
            saving = self.saving

        self.old_position = deepcopy(self.new_position)
        self.steps += 1
        action = np.reshape(action, [self.dim_action])

        current_action_1 = action[0]
        current_action_2 = action[1]

        self.new_position[0] = self.old_position[0] + np.clip(
            self.speed
            * np.cos(self.orientation)
            * (current_action_1 + current_action_2)
            / 2,
            -self.speed,
            self.speed,
        )
        self.new_position[1] = self.old_position[1] + np.clip(
            self.speed
            * np.sin(self.orientation)
            * (current_action_1 + current_action_2)
            / 2,
            -self.speed,
            self.speed,
        )

        self.orientation += np.clip(
            (current_action_1 - current_action_2) / 2 / self.radius, -np.pi, np.pi
        )
        while self.orientation >= 2 * np.pi:
            self.orientation -= 2 * np.pi
        while self.orientation < 0:
            self.orientation += 2 * np.pi

        self.reward = self.reward_fun()
        self.reward_signal = self.reward

        if (
            self.new_position[0] > 15
            or self.new_position[0] < 0
            or self.new_position[1] > 15
            or self.new_position[1] < 0
        ):
            self.new_position = self.old_position

        if self.steps >= self.max_steps:
            self.done = 1

        if saving and self.saving:
            self.savelog(self.reward, self.done)

        return self.get_obs(), self.reward, self.done, {}

    def savelog(self, r, d):
        if self.saving:
            self.file_pointer.write("%f, %f \n" % (r, d))
        else:
            warnings.warn("cannot save!")

    def render(self, mode="human"):
        screen_width = 500
        screen_height = 500

        world_width = 15
        scale = screen_width / world_width
        target_radius = 0.3
        car_radius = 0.1

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(0, 15, 0, 15)
            self.red = rendering.make_circle(target_radius)
            self.green = rendering.make_circle(target_radius)
            self.blue = rendering.make_circle(target_radius)
            self.red.set_color(1.0, 0.0, 0.0)
            self.green.set_color(0.0, 1.0, 0.0)
            self.blue.set_color(0.0, 0.0, 1.0)
            self.red_trans = rendering.Transform()
            self.green_trans = rendering.Transform()
            self.blue_trans = rendering.Transform()
            self.red.add_attr(self.red_trans)
            self.green.add_attr(self.green_trans)
            self.blue.add_attr(self.blue_trans)
            self.viewer.add_geom(self.red)
            self.viewer.add_geom(self.green)
            self.viewer.add_geom(self.blue)

            self.car = rendering.make_circle(car_radius)
            self.car.set_color(0.0, 0.0, 0.0)
            self.car_trans = rendering.Transform()
            self.car.add_attr(self.car_trans)
            self.viewer.add_geom(self.car)

            self.car_orientation = rendering.make_polygon(
                [(0, 0.2), (0, -0.2), (0.4, 0)]
            )
            self.car_orientation.set_color(1.0, 0.0, 1.0)
            self.rotation = rendering.Transform()
            self.car_orientation.add_attr(self.rotation)
            self.viewer.add_geom(self.car_orientation)

        self.red_trans.set_translation(self.red_position[0], self.red_position[1])
        self.green_trans.set_translation(self.green_position[0], self.green_position[1])
        self.blue_trans.set_translation(self.blue_position[0], self.blue_position[1])
        self.car_trans.set_translation(self.new_position[0], self.new_position[1])
        self.rotation.set_translation(self.new_position[0], self.new_position[1])
        self.rotation.set_rotation(self.orientation)

        return self.viewer.render(return_rgb_array=(mode == "rgb_array"))

    def getWindow(self):
        return self.viewer

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class PendulumV:
    """
    Partially observed Pendulum:
    Only the current angular velocity is observed
    """

    def __init__(self):

        self.env = gym.make("Pendulum-v0")
        self.red_position = 0
        self.green_position = 0
        self.blue_position = 0

        self.action_space = spaces.Box(low=-2, high=2, shape=(1,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,))

    def reset(self):
        obs = self.env.reset()
        self.old_position = obs[2:3]
        return obs[2:3]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        self.new_position = obs[2:3]
        self.old_position = deepcopy(self.new_position)
        return obs[2:3], r, done, {}

    def render(self, mode="human"):
        self.env.render()


class CartPoleP(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([self.x_threshold * 2, self.theta_threshold_radians * 2])

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        if not self.action_space.contains(action):
            warnings.warn("%r (%s) invalid" % (action, type(action)))

        action = action.clip(self.action_space.low, self.action_space.high)

        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    """
                            You are calling 'step()' even though this environment has already returned
                            done = True. You should always call 'reset()' once you receive 'done = True'
                            Any further steps are undefined behavior.
                            """
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state)[0::2], reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)[0::2]

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == "rgb_array"))

    def close(self):
        if self.viewer:
            self.viewer.close()


class CartPoleV(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        if not self.action_space.contains(action):
            warnings.warn("%r (%s) invalid" % (action, type(action)))

        action = action.clip(self.action_space.low, self.action_space.high)
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    """
                            You are calling 'step()' even though this environment has already returned
                            done = True. You should always call 'reset()' once you receive 'done = True'
                            Any further steps are undefined behavior.
                            """
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state)[1::2], reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)[1::2]

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == "rgb_array"))

    def close(self):
        if self.viewer:
            self.viewer.close()


class HopperP(gym.Env):
    """
    Partially observed Hopper:
    Only the current joint angles and positions are observed
    """

    def __init__(self):

        self.env = gym.make("Hopper-v2")

        self.nq = 5

        self.action_space = self.env.action_space

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nq,))

    def reset(self):
        obs = self.env.reset()
        return obs[: self.nq]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        return obs[: self.nq], r, done, {}

    def render(self, mode="human"):
        self.env.render()


class HopperV(gym.Env):
    """
    Partially observed Hopper:
    Only the current joint velocities and velocities are observed
    """

    def __init__(self):

        self.env = gym.make("Hopper-v2")

        self.nv = 6

        self.action_space = self.env.action_space

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nv,))

    def reset(self):
        obs = self.env.reset()
        return obs[-self.nv :]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        return obs[-self.nv :], r, done, {}

    def render(self, mode="human"):
        self.env.render()


class RsHopperP(gym.Env):
    """
    Partially observed RoboschoolHopper:
    Only the current joint angles and positions are observed
    """

    def __init__(self):

        self.env = gym.make("RoboschoolHopper-v1")

        self.nq = 9

        self.action_space = self.env.action_space

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nq,))

    def reset(self):
        obs = self.env.reset()
        return obs[[0, 1, 2, 6, 7, 8, 10, 12, 14]]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        return obs[[0, 1, 2, 6, 7, 8, 10, 12, 14]], r, done, {}

    def render(self, mode="human"):
        self.env.render()


class RsHopperV(gym.Env):
    """
    Partially observed RoboschoolHopper:
    Only the current velocities and joint velocities are observed
    """

    def __init__(self):

        self.env = gym.make("RoboschoolHopper-v1")

        self.nq = 6

        self.action_space = self.env.action_space

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nq,))

    def reset(self):
        obs = self.env.reset()
        return obs[[3, 4, 5, 9, 11, 13]]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        return obs[[3, 4, 5, 9, 11, 13]], r, done, {}

    def render(self, mode="human"):
        self.env.render()


class RsAntV(gym.Env):
    """
    Partially observed RoboschoolAnt:
    Only the current velocities and joint velocities are observed
    """

    def __init__(self):

        self.env = gym.make("RoboschoolAnt-v1")

        self.nq = 3 + 8

        self.action_space = self.env.action_space

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nq,))

    def reset(self):
        obs = self.env.reset()
        return obs[[3, 4, 5, 9, 11, 13, 15, 17, 19, 21, 23]]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        return obs[[3, 4, 5, 9, 11, 13, 15, 17, 19, 21, 23]], r, done, {}

    def render(self, mode="human"):
        self.env.render()


class RsAntP(gym.Env):
    """
    Partially observed RoboschoolAnt:
    Only the current joint angles and positions are observed
    """

    def __init__(self):

        self.env = gym.make("RoboschoolAnt-v1")

        self.nq = 5 + 8 + 4

        self.action_space = self.env.action_space

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nq,))

    def reset(self):
        obs = self.env.reset()
        return obs[[0, 1, 2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 26, 27]]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        return (
            obs[[0, 1, 2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 26, 27]],
            r,
            done,
            {},
        )

    def render(self, mode="human"):
        self.env.render()


class RsHalfCheetahP(gym.Env):
    """
    Partially observed RoboschoolHalfCheetah:
    Only the current joint angles and positions are observed
    """

    def __init__(self):

        self.env = gym.make("RoboschoolHalfCheetah-v1")

        self.nq = 5 + 6 + 6

        self.action_space = self.env.action_space

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nq,))

    def reset(self):
        obs = self.env.reset()
        return obs[[0, 1, 2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25]]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        return (
            obs[[0, 1, 2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25]],
            r,
            done,
            {},
        )

    def render(self, mode="human"):
        self.env.render()


class RsHalfCheetahV(gym.Env):
    """
    Partially observed RoboschoolHalfCheetah:
    Only the current velocities and joint velocities are observed
    """

    def __init__(self):

        self.env = gym.make("RoboschoolHalfCheetah-v1")

        self.nq = 3 + 6

        self.action_space = self.env.action_space

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nq,))

    def reset(self):
        obs = self.env.reset()
        return obs[[3, 4, 5, 9, 11, 13, 15, 17, 19]]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        return obs[[3, 4, 5, 9, 11, 13, 15, 17, 19]], r, done, {}

    def render(self, mode="human"):
        self.env.render()


class RsHumanoidP(gym.Env):
    """
    Partially observed RoboschoolHumanoid:
    Only the current joint angles and positions are observed
    """

    def __init__(self):
        self.env = gym.make("RoboschoolHumanoid-v1")

        self.nq = 5 + 17 + 2

        self.action_space = self.env.action_space

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nq,))

    def reset(self):
        obs = self.env.reset()
        return obs[
            [
                0,
                1,
                2,
                6,
                7,
                8,
                10,
                12,
                14,
                16,
                18,
                20,
                22,
                24,
                26,
                28,
                30,
                32,
                34,
                36,
                38,
                40,
                42,
                43,
            ]
        ]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        return (
            obs[
                [
                    0,
                    1,
                    2,
                    6,
                    7,
                    8,
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
                    30,
                    32,
                    34,
                    36,
                    38,
                    40,
                    42,
                    43,
                ]
            ],
            r,
            done,
            {},
        )

    def render(self, mode="human"):
        self.env.render()


class RsHumanoidV(gym.Env):
    """
    Partially observed RoboschoolHumanoid:
    Only the current velocities and joint velocities are observed
    """

    def __init__(self):
        self.env = gym.make("RoboschoolHumanoid-v1")

        self.nq = 3 + 17

        self.action_space = self.env.action_space

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nq,))

    def reset(self):
        obs = self.env.reset()
        return obs[
            [3, 4, 5, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]
        ]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        return (
            obs[
                [
                    3,
                    4,
                    5,
                    9,
                    11,
                    13,
                    15,
                    17,
                    19,
                    21,
                    23,
                    25,
                    27,
                    29,
                    31,
                    33,
                    35,
                    37,
                    39,
                    41,
                ]
            ],
            r,
            done,
            {},
        )

    def render(self, mode="human"):
        self.env.render()


class RsWalker2dP(gym.Env):
    """
    Partially observed RoboschoolWalker2d:
    Only the current joint angles and positions are observed
    """

    def __init__(self):
        self.env = gym.make("RoboschoolWalker2d-v1")

        self.nq = 5 + 6 + 2

        self.action_space = self.env.action_space

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nq,))

    def reset(self):
        obs = self.env.reset()
        return obs[[0, 1, 2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 21]]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        return obs[[0, 1, 2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 21]], r, done, {}

    def render(self, mode="human"):
        self.env.render()


class RsWalker2dV(gym.Env):
    """
    Partially observed RoboschoolWalker2d:
    Only the current velocities and joint velocities are observed
    """

    def __init__(self):
        self.env = gym.make("RoboschoolWalker2d-v1")

        self.nq = 3 + 6

        self.action_space = self.env.action_space

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nq,))

    def reset(self):
        obs = self.env.reset()
        return obs[[3, 4, 5, 9, 11, 13, 15, 17, 19]]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        return obs[[3, 4, 5, 9, 11, 13, 15, 17, 19]], r, done, {}

    def render(self, mode="human"):
        self.env.render()
