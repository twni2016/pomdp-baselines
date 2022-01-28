import os
import time

import gym
import numpy as np
import torch

import utils.config_utils as config_utl
from algorithms.dqn import DQN
from algorithms.sac import SAC
from environments.make_env import make_env
from utils import helpers as utl, offline_utils as off_utl
from torchkit import pytorch_utils as ptu
from torchkit.networks import FlattenMlp
from data_management.storage_policy import MultiTaskPolicyStorage
from utils.tb_logger import TBLogger
from models.policy import TanhGaussianPolicy


class Learner:
    """
    Learner class.
    """

    def __init__(self, args):
        """
        Seeds everything.
        Initialises: logger, environments, policy (+storage +optimiser).
        """

        self.args = args

        # make sure everything has the same seed
        utl.seed(self.args.seed)

        # initialise environment
        self.env = make_env(
            self.args.env_name,
            self.args.max_rollouts_per_task,
            seed=self.args.seed,
            n_tasks=1,
            modify_init_state_dist=self.args.modify_init_state_dist
            if "modify_init_state_dist" in self.args
            else False,
            on_circle_init_state=self.args.on_circle_init_state
            if "on_circle_init_state" in self.args
            else True,
        )

        # saving buffer with task in name folder
        if hasattr(self.args, "save_buffer") and self.args.save_buffer:
            env_dir = os.path.join(
                self.args.main_save_dir, "{}".format(self.args.env_name)
            )
            goal = self.env.unwrapped._goal
            self.output_dir = os.path.join(
                env_dir,
                self.args.save_dir,
                "seed_{}_".format(self.args.seed)
                + off_utl.create_goal_path_ext_from_goal(goal),
            )

        if self.args.save_models or self.args.save_buffer:
            os.makedirs(self.output_dir, exist_ok=True)
            config_utl.save_config_file(args, self.output_dir)

        # initialize tensorboard logger
        if self.args.log_tensorboard:
            self.tb_logger = TBLogger(self.args)

        # if not self.args.log_tensorboard:
        #     self.save_config_json_file()
        # unwrapped env to get some info about the environment
        unwrapped_env = self.env.unwrapped

        # calculate what the maximum length of the trajectories is
        args.max_trajectory_len = unwrapped_env._max_episode_steps
        args.max_trajectory_len *= self.args.max_rollouts_per_task
        self.args.max_trajectory_len = args.max_trajectory_len

        # get action / observation dimensions
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.args.action_dim = 1
        else:
            self.args.action_dim = self.env.action_space.shape[0]
        self.args.obs_dim = self.env.observation_space.shape[0]
        self.args.num_states = (
            unwrapped_env.num_states if hasattr(unwrapped_env, "num_states") else None
        )
        self.args.act_space = self.env.action_space

        # simulate env step to get reward types
        _, _, _, info = unwrapped_env.step(unwrapped_env.action_space.sample())
        reward_types = [
            reward_type
            for reward_type in list(info.keys())
            if reward_type.startswith("reward")
        ]

        # support dense rewards training (if exists)
        self.args.dense_train_sparse_test = (
            self.args.dense_train_sparse_test
            if "dense_train_sparse_test" in self.args
            else False
        )

        # initialize policy
        self.initialize_policy()
        # initialize buffer for RL updates
        self.policy_storage = MultiTaskPolicyStorage(
            max_replay_buffer_size=int(self.args.policy_buffer_size),
            obs_dim=self.args.obs_dim,
            action_space=self.env.action_space,
            tasks=[0],
            trajectory_len=args.max_trajectory_len,
            num_reward_arrays=len(reward_types)
            if reward_types and self.args.dense_train_sparse_test
            else 1,
            reward_types=reward_types,
        )

        self.args.belief_reward = False  # initialize arg to not use belief rewards

    def initialize_policy(self):

        if self.args.policy == "dqn":
            assert (
                self.args.act_space.__class__.__name__ == "Discrete"
            ), "Can't train DQN with continuous action space!"
            q_network = FlattenMlp(
                input_size=self.args.obs_dim,
                output_size=self.args.act_space.n,
                hidden_sizes=self.args.dqn_layers,
            )
            self.agent = DQN(
                q_network,
                # optimiser_vae=self.optimizer_vae,
                lr=self.args.policy_lr,
                gamma=self.args.gamma,
                eps_init=self.args.dqn_epsilon_init,
                eps_final=self.args.dqn_epsilon_final,
                exploration_iters=self.args.dqn_exploration_iters,
                tau=self.args.soft_target_tau,
            ).to(ptu.device)
        # elif self.args.policy == 'ddqn':
        #     assert self.args.act_space.__class__.__name__ == "Discrete", (
        #         "Can't train DDQN with continuous action space!")
        #     q_network = FlattenMlp(input_size=self.args.obs_dim,
        #                            output_size=self.args.act_space.n,
        #                            hidden_sizes=self.args.dqn_layers)
        #     self.agent = DoubleDQN(
        #         q_network,
        #         # optimiser_vae=self.optimizer_vae,
        #         lr=self.args.policy_lr,
        #         eps_optim=self.args.dqn_eps,
        #         alpha_optim=self.args.dqn_alpha,
        #         gamma=self.args.gamma,
        #         eps_init=self.args.dqn_epsilon_init,
        #         eps_final=self.args.dqn_epsilon_final,
        #         exploration_iters=self.args.dqn_exploration_iters,
        #         tau=self.args.soft_target_tau,
        #     ).to(ptu.device)
        elif self.args.policy == "sac":
            assert (
                self.args.act_space.__class__.__name__ == "Box"
            ), "Can't train SAC with discrete action space!"
            q1_network = FlattenMlp(
                input_size=self.args.obs_dim + self.args.action_dim,
                output_size=1,
                hidden_sizes=self.args.dqn_layers,
            )
            q2_network = FlattenMlp(
                input_size=self.args.obs_dim + self.args.action_dim,
                output_size=1,
                hidden_sizes=self.args.dqn_layers,
            )
            policy = TanhGaussianPolicy(
                obs_dim=self.args.obs_dim,
                action_dim=self.args.action_dim,
                hidden_sizes=self.args.policy_layers,
            )
            self.agent = SAC(
                policy,
                q1_network,
                q2_network,
                actor_lr=self.args.actor_lr,
                critic_lr=self.args.critic_lr,
                gamma=self.args.gamma,
                tau=self.args.soft_target_tau,
                entropy_alpha=self.args.entropy_alpha,
                automatic_entropy_tuning=self.args.automatic_entropy_tuning,
                alpha_lr=self.args.alpha_lr,
            ).to(ptu.device)
        else:
            raise NotImplementedError

    def train(self):
        """
        meta-training loop
        """

        self._start_training()
        self.task_idx = 0
        for iter_ in range(self.args.num_iters):
            self.training_mode(True)
            if iter_ == 0:
                print("Collecting initial pool of data..")
                self.env.reset_task(idx=self.task_idx)
                self.collect_rollouts(
                    num_rollouts=self.args.num_init_rollouts_pool, random_actions=True
                )
                print("Done!")
            # collect data from subset of train tasks

            self.env.reset_task(idx=self.task_idx)
            self.collect_rollouts(num_rollouts=self.args.num_rollouts_per_iter)
            # update
            train_stats = self.update([self.task_idx])
            self.training_mode(False)

            if self.args.policy == "dqn":
                self.agent.set_exploration_parameter(iter_ + 1)
            # evaluate and log
            if (iter_ + 1) % self.args.log_interval == 0:
                self.log(iter_ + 1, train_stats)

    def update(self, tasks):
        """
        RL updates
        :param tasks: list/array of task indices. perform update based on the tasks
        :return:
        """

        # --- RL TRAINING ---
        rl_losses_agg = {}
        for update in range(self.args.rl_updates_per_iter):
            # sample random RL batch
            obs, actions, rewards, next_obs, terms = self.sample_rl_batch(
                tasks, self.args.batch_size
            )
            # flatten out task dimension
            t, b, _ = obs.size()
            obs = obs.view(t * b, -1)
            actions = actions.view(t * b, -1)
            rewards = rewards.view(t * b, -1)
            next_obs = next_obs.view(t * b, -1)
            terms = terms.view(t * b, -1)

            # RL update
            rl_losses = self.agent.update(obs, actions, rewards, next_obs, terms)

            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # take mean
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += self.args.rl_updates_per_iter

        return rl_losses_agg

    def evaluate(self, tasks):
        num_episodes = self.args.max_rollouts_per_task
        num_steps_per_episode = self.env.unwrapped._max_episode_steps

        returns_per_episode = np.zeros((len(tasks), num_episodes))
        success_rate = np.zeros(len(tasks))

        if self.args.policy == "dqn":
            values = np.zeros((len(tasks), self.args.max_trajectory_len))
        else:
            obs_size = self.env.unwrapped.observation_space.shape[0]
            observations = np.zeros(
                (len(tasks), self.args.max_trajectory_len + 1, obs_size)
            )
            log_probs = np.zeros((len(tasks), self.args.max_trajectory_len))

        for task_idx, task in enumerate(tasks):

            obs = ptu.from_numpy(self.env.reset(task))
            obs = obs.reshape(-1, obs.shape[-1])
            step = 0

            if self.args.policy == "sac":
                observations[task_idx, step, :] = ptu.get_numpy(obs[0, :obs_size])

            for episode_idx in range(num_episodes):
                running_reward = 0.0
                for step_idx in range(num_steps_per_episode):
                    # add distribution parameters to observation - policy is conditioned on posterior
                    if self.args.policy == "dqn":
                        action, value = self.agent.act(obs=obs, deterministic=True)
                    else:
                        action, _, _, log_prob = self.agent.act(
                            obs=obs,
                            deterministic=self.args.eval_deterministic,
                            return_log_prob=True,
                        )
                    # observe reward and next obs
                    next_obs, reward, done, info = utl.env_step(
                        self.env, action.squeeze(dim=0)
                    )
                    running_reward += reward.item()
                    if self.args.policy == "dqn":
                        values[task_idx, step] = value.item()
                    else:
                        observations[task_idx, step + 1, :] = ptu.get_numpy(
                            next_obs[0, :obs_size]
                        )
                        log_probs[task_idx, step] = ptu.get_numpy(log_prob[0])

                    if (
                        "is_goal_state" in dir(self.env.unwrapped)
                        and self.env.unwrapped.is_goal_state()
                    ):
                        success_rate[task_idx] = 1.0
                    # set: obs <- next_obs
                    obs = next_obs.clone()
                    step += 1

                returns_per_episode[task_idx, episode_idx] = running_reward

        if self.args.policy == "dqn":
            return returns_per_episode, success_rate, values
        else:
            return returns_per_episode, success_rate, log_probs, observations

    def log(self, iteration, train_stats):
        # --- save models ---
        if iteration % self.args.save_interval == 0:
            if self.args.save_models:
                if self.args.log_tensorboard:
                    save_path = os.path.join(
                        self.tb_logger.full_output_folder, "models"
                    )
                else:
                    save_path = os.path.join(self.output_dir, "models")
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                torch.save(
                    self.agent.state_dict(),
                    os.path.join(save_path, "agent{0}.pt".format(iteration)),
                )
            if hasattr(self.args, "save_buffer") and self.args.save_buffer:
                self.save_buffer()
        # evaluate to get more stats
        if self.args.policy == "dqn":
            # get stats on train tasks
            returns_train, success_rate_train, values = self.evaluate([0])
        else:
            # get stats on train tasks
            returns_train, success_rate_train, log_probs, observations = self.evaluate(
                [0]
            )

        if self.args.log_tensorboard:
            if self.args.policy != "dqn":
                #     self.env.reset(0)
                #     self.tb_logger.writer.add_figure('policy_vis_train/task_0',
                #                                      utl_eval.plot_rollouts(observations[0, :], self.env),
                #                                      self._n_env_steps_total)
                #     obs, _, _, _, _ = self.sample_rl_batch(tasks=[0],
                #                                            batch_size=self.policy_storage.task_buffers[0].size())
                #     self.tb_logger.writer.add_figure('state_space_coverage/task_0',
                #                                      utl_eval.plot_visited_states(ptu.get_numpy(obs[0][:, :2]), self.env),
                #                                      self._n_env_steps_total)
                pass
            # some metrics
            self.tb_logger.writer.add_scalar(
                "metrics/successes_in_buffer",
                self._successes_in_buffer / self._n_env_steps_total,
                self._n_env_steps_total,
            )

            if self.args.max_rollouts_per_task > 1:
                for episode_idx in range(self.args.max_rollouts_per_task):
                    self.tb_logger.writer.add_scalar(
                        "returns_multi_episode/episode_{}".format(episode_idx + 1),
                        np.mean(returns_train[:, episode_idx]),
                        self._n_env_steps_total,
                    )
                self.tb_logger.writer.add_scalar(
                    "returns_multi_episode/sum",
                    np.mean(np.sum(returns_train, axis=-1)),
                    self._n_env_steps_total,
                )
                self.tb_logger.writer.add_scalar(
                    "returns_multi_episode/success_rate",
                    np.mean(success_rate_train),
                    self._n_env_steps_total,
                )
            else:
                self.tb_logger.writer.add_scalar(
                    "returns/returns_mean_train",
                    np.mean(returns_train),
                    self._n_env_steps_total,
                )
                self.tb_logger.writer.add_scalar(
                    "returns/returns_std_train",
                    np.std(returns_train),
                    self._n_env_steps_total,
                )
                self.tb_logger.writer.add_scalar(
                    "returns/success_rate_train",
                    np.mean(success_rate_train),
                    self._n_env_steps_total,
                )
            # policy
            if self.args.policy == "dqn":
                self.tb_logger.writer.add_scalar(
                    "policy/value_init", np.mean(values[:, 0]), self._n_env_steps_total
                )
                self.tb_logger.writer.add_scalar(
                    "policy/value_halfway",
                    np.mean(values[:, int(values.shape[-1] / 2)]),
                    self._n_env_steps_total,
                )
                self.tb_logger.writer.add_scalar(
                    "policy/value_final",
                    np.mean(values[:, -1]),
                    self._n_env_steps_total,
                )

                self.tb_logger.writer.add_scalar(
                    "policy/exploration_epsilon",
                    self.agent.eps,
                    self._n_env_steps_total,
                )
                # RL losses
                self.tb_logger.writer.add_scalar(
                    "rl_losses/qf_loss_vs_n_updates",
                    train_stats["qf_loss"],
                    self._n_rl_update_steps_total,
                )
                self.tb_logger.writer.add_scalar(
                    "rl_losses/qf_loss_vs_n_env_steps",
                    train_stats["qf_loss"],
                    self._n_env_steps_total,
                )
            else:
                self.tb_logger.writer.add_scalar(
                    "policy/log_prob", np.mean(log_probs), self._n_env_steps_total
                )
                self.tb_logger.writer.add_scalar(
                    "rl_losses/qf1_loss",
                    train_stats["qf1_loss"],
                    self._n_env_steps_total,
                )
                self.tb_logger.writer.add_scalar(
                    "rl_losses/qf2_loss",
                    train_stats["qf2_loss"],
                    self._n_env_steps_total,
                )
                self.tb_logger.writer.add_scalar(
                    "rl_losses/policy_loss",
                    train_stats["policy_loss"],
                    self._n_env_steps_total,
                )
                self.tb_logger.writer.add_scalar(
                    "rl_losses/alpha_entropy_loss",
                    train_stats["alpha_entropy_loss"],
                    self._n_env_steps_total,
                )

            # weights and gradients
            if self.args.policy == "dqn":
                self.tb_logger.writer.add_scalar(
                    "weights/q_network",
                    list(self.agent.qf.parameters())[0].mean(),
                    self._n_env_steps_total,
                )
                if list(self.agent.qf.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf.parameters())
                    self.tb_logger.writer.add_scalar(
                        "gradients/q_network",
                        sum(
                            [param_list[i].grad.mean() for i in range(len(param_list))]
                        ),
                        self._n_env_steps_total,
                    )
                self.tb_logger.writer.add_scalar(
                    "weights/q_target",
                    list(self.agent.target_qf.parameters())[0].mean(),
                    self._n_env_steps_total,
                )
                if list(self.agent.target_qf.parameters())[0].grad is not None:
                    param_list = list(self.agent.target_qf.parameters())
                    self.tb_logger.writer.add_scalar(
                        "gradients/q_target",
                        sum(
                            [param_list[i].grad.mean() for i in range(len(param_list))]
                        ),
                        self._n_env_steps_total,
                    )
            else:
                self.tb_logger.writer.add_scalar(
                    "weights/q1_network",
                    list(self.agent.qf1.parameters())[0].mean(),
                    self._n_env_steps_total,
                )
                if list(self.agent.qf1.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf1.parameters())
                    self.tb_logger.writer.add_scalar(
                        "gradients/q1_network",
                        sum(
                            [param_list[i].grad.mean() for i in range(len(param_list))]
                        ),
                        self._n_env_steps_total,
                    )
                self.tb_logger.writer.add_scalar(
                    "weights/q1_target",
                    list(self.agent.qf1_target.parameters())[0].mean(),
                    self._n_env_steps_total,
                )
                if list(self.agent.qf1_target.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf1_target.parameters())
                    self.tb_logger.writer.add_scalar(
                        "gradients/q1_target",
                        sum(
                            [param_list[i].grad.mean() for i in range(len(param_list))]
                        ),
                        self._n_env_steps_total,
                    )
                self.tb_logger.writer.add_scalar(
                    "weights/q2_network",
                    list(self.agent.qf2.parameters())[0].mean(),
                    self._n_env_steps_total,
                )
                if list(self.agent.qf2.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf2.parameters())
                    self.tb_logger.writer.add_scalar(
                        "gradients/q2_network",
                        sum(
                            [param_list[i].grad.mean() for i in range(len(param_list))]
                        ),
                        self._n_env_steps_total,
                    )
                self.tb_logger.writer.add_scalar(
                    "weights/q2_target",
                    list(self.agent.qf2_target.parameters())[0].mean(),
                    self._n_env_steps_total,
                )
                if list(self.agent.qf2_target.parameters())[0].grad is not None:
                    param_list = list(self.agent.qf2_target.parameters())
                    self.tb_logger.writer.add_scalar(
                        "gradients/q2_target",
                        sum(
                            [param_list[i].grad.mean() for i in range(len(param_list))]
                        ),
                        self._n_env_steps_total,
                    )
                self.tb_logger.writer.add_scalar(
                    "weights/policy",
                    list(self.agent.policy.parameters())[0].mean(),
                    self._n_env_steps_total,
                )
                if list(self.agent.policy.parameters())[0].grad is not None:
                    param_list = list(self.agent.policy.parameters())
                    self.tb_logger.writer.add_scalar(
                        "gradients/policy",
                        sum(
                            [param_list[i].grad.mean() for i in range(len(param_list))]
                        ),
                        self._n_env_steps_total,
                    )

        print(
            "Iteration -- {}, Success rate -- {:.3f}, Avg. return -- {:.3f}, Elapsed time {:5d}[s]".format(
                iteration,
                np.mean(success_rate_train),
                np.mean(np.sum(returns_train, axis=-1)),
                int(time.time() - self._start_time),
            )
        )
        # output to user
        # print("Iteration -- {:3d}, Num. RL updates -- {:6d}, Elapsed time {:5d}[s]".
        #       format(iteration,
        #              self._n_rl_update_steps_total,
        #              int(time.time() - self._start_time)))

    def training_mode(self, mode):
        self.agent.train(mode)

    def collect_rollouts(self, num_rollouts, random_actions=False):
        """

        :param num_rollouts:
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        :return:
        """

        for rollout in range(num_rollouts):
            obs = ptu.from_numpy(self.env.reset(self.task_idx))
            obs = obs.reshape(-1, obs.shape[-1])
            done_rollout = False

            while not done_rollout:
                if random_actions:
                    if self.args.policy == "dqn":
                        action = ptu.FloatTensor(
                            [[[self.env.action_space.sample()]]]
                        ).long()  # Sample random action
                    else:
                        action = ptu.FloatTensor(
                            [self.env.action_space.sample()]
                        )  # Sample random action
                else:
                    if self.args.policy == "dqn":
                        action, _ = self.agent.act(obs=obs)  # DQN
                    else:
                        action, _, _, _ = self.agent.act(obs=obs)  # SAC
                # observe reward and next obs
                next_obs, reward, done, info = utl.env_step(
                    self.env, action.squeeze(dim=0)
                )
                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                # add data to policy buffer - (s+, a, r, s'+, term)
                term = (
                    self.env.unwrapped.is_goal_state()
                    if "is_goal_state" in dir(self.env.unwrapped)
                    else False
                )
                if self.args.dense_train_sparse_test:
                    rew_to_buffer = {
                        rew_type: rew
                        for rew_type, rew in info.items()
                        if rew_type.startswith("reward")
                    }
                else:
                    rew_to_buffer = ptu.get_numpy(reward.squeeze(dim=0))
                self.policy_storage.add_sample(
                    task=self.task_idx,
                    observation=ptu.get_numpy(obs.squeeze(dim=0)),
                    action=ptu.get_numpy(action.squeeze(dim=0)),
                    reward=rew_to_buffer,
                    terminal=np.array([term], dtype=float),
                    next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                )

                # set: obs <- next_obs
                obs = next_obs.clone()

                # update statistics
                self._n_env_steps_total += 1
                if (
                    "is_goal_state" in dir(self.env.unwrapped)
                    and self.env.unwrapped.is_goal_state()
                ):  # count successes
                    self._successes_in_buffer += 1
            self._n_rollouts_total += 1

    def sample_rl_batch(self, tasks, batch_size):
        """sample batch of unordered rl training data from a list/array of tasks"""
        # this batch consists of transitions sampled randomly from replay buffer
        batches = [
            ptu.np_to_pytorch_batch(self.policy_storage.random_batch(task, batch_size))
            for task in tasks
        ]
        unpacked = [utl.unpack_batch(batch) for batch in batches]
        # group elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_rl_update_steps_total = 0
        self._n_vae_update_steps_total = 0
        self._n_rollouts_total = 0
        self._successes_in_buffer = 0

        self._start_time = time.time()

    def load_model(self, device="cpu", **kwargs):
        if "agent_path" in kwargs:
            self.agent.load_state_dict(
                torch.load(kwargs["agent_path"], map_location=device)
            )
        self.training_mode(False)

    def save_buffer(self):
        size = self.policy_storage.task_buffers[0].size()
        np.save(
            os.path.join(self.output_dir, "obs"),
            self.policy_storage.task_buffers[0]._observations[:size],
        )
        np.save(
            os.path.join(self.output_dir, "actions"),
            self.policy_storage.task_buffers[0]._actions[:size],
        )
        if self.args.dense_train_sparse_test:
            for reward_type, reward_arr in self.policy_storage.task_buffers[
                0
            ]._rewards.items():
                np.save(os.path.join(self.output_dir, reward_type), reward_arr[:size])
        else:
            np.save(
                os.path.join(self.output_dir, "rewards"),
                self.policy_storage.task_buffers[0]._rewards[:size],
            )
        np.save(
            os.path.join(self.output_dir, "next_obs"),
            self.policy_storage.task_buffers[0]._next_obs[:size],
        )
        np.save(
            os.path.join(self.output_dir, "terminals"),
            self.policy_storage.task_buffers[0]._terminals[:size],
        )
