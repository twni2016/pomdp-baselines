import numpy as np
import random
import matplotlib.pyplot as plt

from environments.mujoco.ant_multitask_base import MultitaskAntEnv


class AntSemiCircleEnv(MultitaskAntEnv):
    def __init__(
        self,
        task={},
        n_tasks=2,
        max_episode_steps=200,
        modify_init_state_dist=False,
        on_circle_init_state=False,
        **kwargs
    ):
        super(AntSemiCircleEnv, self).__init__(task, n_tasks, **kwargs)
        # self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.modify_init_state_dist = modify_init_state_dist
        self.on_circle_init_state = on_circle_init_state

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(
            np.abs(xposafter[:2] - self._goal)
        )  # make it happy, not suicidal
        # goal_reward = -(np.sum((xposafter[:2] - self._goal) ** 2) ** 0.5)

        ctrl_cost = 0.1 * np.square(action).sum()
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # survive_reward = 0.0
        # reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        # reward = goal_reward - ctrl_cost - contact_cost
        reward = goal_reward - ctrl_cost
        # reward = goal_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_goal=goal_reward,
                reward_ctrl=-ctrl_cost,
                # reward_contact=-contact_cost,
                task=self._task,
            ),
        )

    def reset_model(self):
        qpos = self.init_qpos
        # just for offline data collection:
        if self.modify_init_state_dist:
            qpos[:2] = np.array(
                [np.random.uniform(-1.5, 1.5), np.random.uniform(-0.5, 1.5)]
            )
            if (
                not self.on_circle_init_state
            ):  # make sure initial state is not on semi-circle
                # while 1 - self.goal_radius <= np.linspace.norm(qpos[:2]) <= 1 + self.goal_radius:
                while (
                    0.8 <= np.linalg.norm(qpos[:2]) <= 1.2
                ):  # TODO: uses privileged knowledge (R=0.2)
                    qpos[:2] = np.array(
                        [np.random.uniform(-1.5, 1.5), np.random.uniform(-0.5, 1.5)]
                    )
        else:
            qpos[:2] = np.array([0, 0])
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reward(self, state, action):
        goal_reward = -np.sum(
            np.abs(state[:2] - self._goal)
        )  # make it happy, not suicidal
        ctrl_cost = 0.1 * np.square(action).sum()
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # reward = goal_reward - ctrl_cost - contact_cost
        reward = goal_reward - ctrl_cost
        return reward

    def set_goal(self, goal):
        self._goal = np.asarray(goal)

    def sample_tasks(self, num_tasks):
        a = np.array([random.uniform(0, np.pi) for _ in range(num_tasks)])
        r = 1
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        tasks = [{"goal": goal} for goal in goals]
        return tasks

    def get_task(self):
        return self._goal

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 90


class SparseAntSemiCircleEnv(AntSemiCircleEnv):
    def __init__(
        self, task={}, n_tasks=2, max_episode_steps=200, goal_radius=0.2, **kwargs
    ):
        self.goal_radius = goal_radius
        super().__init__(task, n_tasks, max_episode_steps, **kwargs)

    def sparsify_rewards(self, d):
        non_goal_reward_keys = []
        for key in d.keys():
            if key.startswith("reward") and key != "reward_goal":
                non_goal_reward_keys.append(key)
        non_goal_rewards = np.sum(
            [d[reward_key] for reward_key in non_goal_reward_keys]
        )
        sparse_goal_reward = 1.0 if self.is_goal_state() else 0.0
        return non_goal_rewards + sparse_goal_reward

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(d)
        return ob, sparse_reward, done, d

    def reward(self, state, action):
        goal_reward = 1.0 if self.is_goal_state(state) else 0.0
        ctrl_cost = 0.1 * np.square(action).sum()
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # reward = goal_reward - ctrl_cost - contact_cost
        reward = goal_reward - ctrl_cost
        return reward

    def is_goal_state(self, state=None):
        if state is None:
            state = np.array(self.get_body_com("torso"))
        if np.linalg.norm(state[:2] - self._goal) <= self.goal_radius:
            return True
        else:
            return False

    def plot_env(self):
        ax = plt.gca()
        # plot half circle and goal position
        angles = np.linspace(0, np.pi, num=100)
        x, y = np.cos(angles), np.sin(angles)
        plt.plot(x, y, color="k")
        # fix visualization
        plt.axis("scaled")
        # ax.set_xlim(-1.25, 1.25)
        ax.set_xlim(-2, 2)
        # ax.set_ylim(-0.25, 1.25)
        ax.set_ylim(-1, 2)
        plt.xticks([])
        plt.yticks([])
        circle = plt.Circle(
            (self._goal[0], self._goal[1]),
            radius=self.goal_radius if hasattr(self, "goal_radius") else 0.1,
            alpha=0.3,
        )
        ax.add_artist(circle)

    def plot_behavior(self, observations, plot_env=True, **kwargs):
        if plot_env:  # whether to plot circle and goal pos..(maybe already exists)
            self.plot_env()
        # visualise behaviour, current position, goal
        plt.plot(observations[:, 0], observations[:, 1], **kwargs)
