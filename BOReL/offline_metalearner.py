import os
import time

import numpy as np
import torch

from algorithms.dqn import DQN
from algorithms.sac import SAC
from environments.make_env import make_env
from utils import helpers as utl, offline_utils as off_utl
from torchkit import pytorch_utils as ptu
from torchkit.networks import FlattenMlp
from data_management.storage_policy import MultiTaskPolicyStorage
from utils import evaluation as utl_eval
from utils.tb_logger import TBLogger
from models.vae import VAE
from models.policy import TanhGaussianPolicy


class OfflineMetaLearner:
    """
    Off-line Meta-Learner class, a.k.a no interaction with env.
    """

    def __init__(self, args):
        """
        Seeds everything.
        Initialises: logger, environments, policy (+storage +optimiser).
        """

        self.args = args

        # make sure everything has the same seed
        utl.seed(self.args.seed)

        # initialize tensorboard logger
        if self.args.log_tensorboard:
            self.tb_logger = TBLogger(self.args)

        self.args, env = off_utl.expand_args(self.args, include_act_space=True)
        if self.args.act_space.__class__.__name__ == "Discrete":
            self.args.policy = "dqn"
        else:
            self.args.policy = "sac"

        # load buffers with data
        if "load_data" not in self.args or self.args.load_data:
            goals, augmented_obs_dim = self.load_buffer(
                env
            )  # env is input just for possible relabelling option
            self.args.augmented_obs_dim = augmented_obs_dim
            self.goals = goals

        # initialize policy
        self.initialize_policy()

        # load vae for inference in evaluation
        self.load_vae()

        # create environment for evaluation
        self.env = make_env(
            args.env_name,
            args.max_rollouts_per_task,
            seed=args.seed,
            n_tasks=self.args.num_eval_tasks,
        )
        if self.args.env_name == "GridNavi-v2":
            self.env.unwrapped.goals = [tuple(goal.astype(int)) for goal in self.goals]

    def initialize_policy(self):
        if self.args.policy == "dqn":
            q_network = FlattenMlp(
                input_size=self.args.augmented_obs_dim,
                output_size=self.args.act_space.n,
                hidden_sizes=self.args.dqn_layers,
            )
            self.agent = DQN(
                q_network,
                # optimiser_vae=self.optimizer_vae,
                lr=self.args.policy_lr,
                gamma=self.args.gamma,
                tau=self.args.soft_target_tau,
            ).to(ptu.device)
        else:
            # assert self.args.act_space.__class__.__name__ == "Box", (
            #     "Can't train SAC with discrete action space!")
            q1_network = FlattenMlp(
                input_size=self.args.augmented_obs_dim + self.args.action_dim,
                output_size=1,
                hidden_sizes=self.args.dqn_layers,
            )
            q2_network = FlattenMlp(
                input_size=self.args.augmented_obs_dim + self.args.action_dim,
                output_size=1,
                hidden_sizes=self.args.dqn_layers,
            )
            policy = TanhGaussianPolicy(
                obs_dim=self.args.augmented_obs_dim,
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
                use_cql=self.args.use_cql if "use_cql" in self.args else False,
                alpha_cql=self.args.alpha_cql if "alpha_cql" in self.args else None,
                entropy_alpha=self.args.entropy_alpha,
                automatic_entropy_tuning=self.args.automatic_entropy_tuning,
                alpha_lr=self.args.alpha_lr,
                clip_grad_value=self.args.clip_grad_value,
            ).to(ptu.device)

    def load_vae(self):
        self.vae = VAE(self.args)
        vae_models_path = os.path.join(
            self.args.vae_dir, self.args.env_name, self.args.vae_model_name, "models"
        )
        off_utl.load_trained_vae(self.vae, vae_models_path)

    def load_buffer(self, env):
        if (
            self.args.hindsight_relabelling
        ):  # without arr_type loading -- GPU will explode
            dataset, goals = off_utl.load_dataset(
                data_dir=self.args.relabelled_data_dir,
                args=self.args,
                num_tasks=self.args.num_train_tasks,
                allow_dense_data_loading=False,
                arr_type="numpy",
            )
            dataset = off_utl.batch_to_trajectories(dataset, self.args)
            dataset, goals = off_utl.mix_task_rollouts(
                dataset, env, goals, self.args
            )  # reward relabelling
            dataset = off_utl.trajectories_to_batch(dataset)
        else:
            dataset, goals = off_utl.load_dataset(
                data_dir=self.args.relabelled_data_dir,
                args=self.args,
                num_tasks=self.args.num_train_tasks,
                allow_dense_data_loading=False,
                arr_type="numpy",
            )
        augmented_obs_dim = dataset[0][0].shape[1]
        self.storage = MultiTaskPolicyStorage(
            max_replay_buffer_size=dataset[0][0].shape[0],
            obs_dim=dataset[0][0].shape[1],
            action_space=self.args.act_space,
            tasks=range(len(goals)),
            trajectory_len=self.args.trajectory_len,
        )
        for task, set in enumerate(dataset):
            self.storage.add_samples(
                task,
                observations=set[0],
                actions=set[1],
                rewards=set[2],
                next_observations=set[3],
                terminals=set[4],
            )
        return goals, augmented_obs_dim

    def train(self):
        self._start_training()
        for iter_ in range(self.args.num_iters):
            self.training_mode(True)
            indices = np.random.choice(len(self.goals), self.args.meta_batch)
            train_stats = self.update(indices)

            self.training_mode(False)
            self.log(iter_ + 1, train_stats)

    def update(self, tasks):
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
            rl_losses = self.agent.update(
                obs,
                actions,
                rewards,
                next_obs,
                terms,
                action_space=self.env.action_space,
            )

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

    def evaluate(self):
        num_episodes = self.args.max_rollouts_per_task
        num_steps_per_episode = self.env.unwrapped._max_episode_steps
        num_tasks = self.args.num_eval_tasks
        obs_size = self.env.unwrapped.observation_space.shape[0]

        returns_per_episode = np.zeros((num_tasks, num_episodes))
        success_rate = np.zeros(num_tasks)

        rewards = np.zeros((num_tasks, self.args.trajectory_len))
        reward_preds = np.zeros((num_tasks, self.args.trajectory_len))
        observations = np.zeros((num_tasks, self.args.trajectory_len + 1, obs_size))
        if self.args.policy == "sac":
            log_probs = np.zeros((num_tasks, self.args.trajectory_len))

        # This part is very specific for the Semi-Circle env
        # if self.args.env_name == 'PointRobotSparse-v0':
        #     reward_belief = np.zeros((num_tasks, self.args.trajectory_len))
        #
        #     low_x, high_x, low_y, high_y = -2., 2., -1., 2.
        #     resolution = 0.1
        #     grid_x = np.arange(low_x, high_x + resolution, resolution)
        #     grid_y = np.arange(low_y, high_y + resolution, resolution)
        #     centers_x = (grid_x[:-1] + grid_x[1:]) / 2
        #     centers_y = (grid_y[:-1] + grid_y[1:]) / 2
        #     yv, xv = np.meshgrid(centers_y, centers_x, sparse=False, indexing='ij')
        #     centers = np.vstack([xv.ravel(), yv.ravel()]).T
        #     n_grid_points = centers.shape[0]
        #     reward_belief_discretized = np.zeros((num_tasks, self.args.trajectory_len, centers.shape[0]))

        for task in self.env.unwrapped.get_all_task_idx():
            obs = ptu.from_numpy(self.env.reset(task))
            obs = obs.reshape(-1, obs.shape[-1])
            step = 0

            # get prior parameters
            with torch.no_grad():
                (
                    task_sample,
                    task_mean,
                    task_logvar,
                    hidden_state,
                ) = self.vae.encoder.prior(batch_size=1)

            observations[task, step, :] = ptu.get_numpy(obs[0, :obs_size])

            for episode_idx in range(num_episodes):
                running_reward = 0.0
                for step_idx in range(num_steps_per_episode):
                    # add distribution parameters to observation - policy is conditioned on posterior
                    augmented_obs = self.get_augmented_obs(obs, task_mean, task_logvar)
                    if self.args.policy == "dqn":
                        action, value = self.agent.act(
                            obs=augmented_obs, deterministic=True
                        )
                    else:
                        action, _, _, log_prob = self.agent.act(
                            obs=augmented_obs,
                            deterministic=self.args.eval_deterministic,
                            return_log_prob=True,
                        )

                    # observe reward and next obs
                    next_obs, reward, done, info = utl.env_step(
                        self.env, action.squeeze(dim=0)
                    )
                    running_reward += reward.item()
                    # done_rollout = False if ptu.get_numpy(done[0][0]) == 0. else True
                    # update encoding
                    (
                        task_sample,
                        task_mean,
                        task_logvar,
                        hidden_state,
                    ) = self.update_encoding(
                        obs=next_obs,
                        action=action,
                        reward=reward,
                        done=done,
                        hidden_state=hidden_state,
                    )
                    rewards[task, step] = reward.item()
                    reward_preds[task, step] = ptu.get_numpy(
                        self.vae.reward_decoder(task_sample, next_obs, obs, action)[
                            0, 0
                        ]
                    )

                    # This part is very specific for the Semi-Circle env
                    # if self.args.env_name == 'PointRobotSparse-v0':
                    #     reward_belief[task, step] = ptu.get_numpy(
                    #         self.vae.compute_belief_reward(task_mean, task_logvar, obs, next_obs, action)[0])
                    #
                    #     reward_belief_discretized[task, step, :] = ptu.get_numpy(
                    #         self.vae.compute_belief_reward(task_mean.repeat(n_grid_points, 1),
                    #                                        task_logvar.repeat(n_grid_points, 1),
                    #                                        None,
                    #                                        torch.cat((ptu.FloatTensor(centers),
                    #                                                   ptu.zeros(centers.shape[0], 1)), dim=-1).unsqueeze(0),
                    #                                        None)[:, 0])

                    observations[task, step + 1, :] = ptu.get_numpy(
                        next_obs[0, :obs_size]
                    )
                    if self.args.policy != "dqn":
                        log_probs[task, step] = ptu.get_numpy(log_prob[0])

                    if (
                        "is_goal_state" in dir(self.env.unwrapped)
                        and self.env.unwrapped.is_goal_state()
                    ):
                        success_rate[task] = 1.0
                    # set: obs <- next_obs
                    obs = next_obs.clone()
                    step += 1

                returns_per_episode[task, episode_idx] = running_reward

        if self.args.policy == "dqn":
            return (
                returns_per_episode,
                success_rate,
                observations,
                rewards,
                reward_preds,
            )
        # This part is very specific for the Semi-Circle env
        # elif self.args.env_name == 'PointRobotSparse-v0':
        #     return returns_per_episode, success_rate, log_probs, observations, \
        #            rewards, reward_preds, reward_belief, reward_belief_discretized, centers
        else:
            return (
                returns_per_episode,
                success_rate,
                log_probs,
                observations,
                rewards,
                reward_preds,
            )

    def log(self, iteration, train_stats):
        # --- save model ---
        if iteration % self.args.save_interval == 0:
            save_path = os.path.join(self.tb_logger.full_output_folder, "models")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(
                self.agent.state_dict(),
                os.path.join(save_path, "agent{0}.pt".format(iteration)),
            )

        if iteration % self.args.log_interval == 0:
            if self.args.policy == "dqn":
                (
                    returns,
                    success_rate,
                    observations,
                    rewards,
                    reward_preds,
                ) = self.evaluate()
            # This part is super specific for the Semi-Circle env
            # elif self.args.env_name == 'PointRobotSparse-v0':
            #     returns, success_rate, log_probs, observations, \
            #     rewards, reward_preds, reward_belief, reward_belief_discretized, points = self.evaluate()
            else:
                (
                    returns,
                    success_rate,
                    log_probs,
                    observations,
                    rewards,
                    reward_preds,
                ) = self.evaluate()

            if self.args.log_tensorboard:
                tasks_to_vis = np.random.choice(self.args.num_eval_tasks, 5)
                for i, task in enumerate(tasks_to_vis):
                    self.env.reset(task)
                    self.tb_logger.writer.add_figure(
                        "policy_vis/task_{}".format(i),
                        utl_eval.plot_rollouts(observations[task, :], self.env),
                        self._n_rl_update_steps_total,
                    )
                    self.tb_logger.writer.add_figure(
                        "reward_prediction_train/task_{}".format(i),
                        utl_eval.plot_rew_pred_vs_rew(
                            rewards[task, :], reward_preds[task, :]
                        ),
                        self._n_rl_update_steps_total,
                    )
                    # self.tb_logger.writer.add_figure('reward_prediction_train/task_{}'.format(i),
                    #                                  utl_eval.plot_rew_pred_vs_reward_belief_vs_rew(rewards[task, :],
                    #                                                                                 reward_preds[task, :],
                    #                                                                                 reward_belief[task, :]),
                    #                                  self._n_rl_update_steps_total)
                    # if self.args.env_name == 'PointRobotSparse-v0':     # This part is super specific for the Semi-Circle env
                    #     for t in range(0, int(self.args.trajectory_len/4), 3):
                    #         self.tb_logger.writer.add_figure('discrete_belief_reward_pred_task_{}/timestep_{}'.format(i, t),
                    #                                          utl_eval.plot_discretized_belief_halfcircle(reward_belief_discretized[task, t, :],
                    #                                                                                      points, self.env,
                    #                                                                                      observations[task, :t+1]),
                    #                                          self._n_rl_update_steps_total)
                if self.args.max_rollouts_per_task > 1:
                    for episode_idx in range(self.args.max_rollouts_per_task):
                        self.tb_logger.writer.add_scalar(
                            "returns_multi_episode/episode_{}".format(episode_idx + 1),
                            np.mean(returns[:, episode_idx]),
                            self._n_rl_update_steps_total,
                        )
                    self.tb_logger.writer.add_scalar(
                        "returns_multi_episode/sum",
                        np.mean(np.sum(returns, axis=-1)),
                        self._n_rl_update_steps_total,
                    )
                    self.tb_logger.writer.add_scalar(
                        "returns_multi_episode/success_rate",
                        np.mean(success_rate),
                        self._n_rl_update_steps_total,
                    )
                else:
                    self.tb_logger.writer.add_scalar(
                        "returns/returns_mean",
                        np.mean(returns),
                        self._n_rl_update_steps_total,
                    )
                    self.tb_logger.writer.add_scalar(
                        "returns/returns_std",
                        np.std(returns),
                        self._n_rl_update_steps_total,
                    )
                    self.tb_logger.writer.add_scalar(
                        "returns/success_rate",
                        np.mean(success_rate),
                        self._n_rl_update_steps_total,
                    )
                if self.args.policy == "dqn":
                    self.tb_logger.writer.add_scalar(
                        "rl_losses/qf_loss_vs_n_updates",
                        train_stats["qf_loss"],
                        self._n_rl_update_steps_total,
                    )
                    self.tb_logger.writer.add_scalar(
                        "weights/q_network",
                        list(self.agent.qf.parameters())[0].mean(),
                        self._n_rl_update_steps_total,
                    )
                    if list(self.agent.qf.parameters())[0].grad is not None:
                        param_list = list(self.agent.qf.parameters())
                        self.tb_logger.writer.add_scalar(
                            "gradients/q_network",
                            sum(
                                [
                                    param_list[i].grad.mean()
                                    for i in range(len(param_list))
                                ]
                            ),
                            self._n_rl_update_steps_total,
                        )
                    self.tb_logger.writer.add_scalar(
                        "weights/q_target",
                        list(self.agent.target_qf.parameters())[0].mean(),
                        self._n_rl_update_steps_total,
                    )
                    if list(self.agent.target_qf.parameters())[0].grad is not None:
                        param_list = list(self.agent.target_qf.parameters())
                        self.tb_logger.writer.add_scalar(
                            "gradients/q_target",
                            sum(
                                [
                                    param_list[i].grad.mean()
                                    for i in range(len(param_list))
                                ]
                            ),
                            self._n_rl_update_steps_total,
                        )
                else:
                    self.tb_logger.writer.add_scalar(
                        "policy/log_prob",
                        np.mean(log_probs),
                        self._n_rl_update_steps_total,
                    )
                    self.tb_logger.writer.add_scalar(
                        "rl_losses/qf1_loss",
                        train_stats["qf1_loss"],
                        self._n_rl_update_steps_total,
                    )
                    self.tb_logger.writer.add_scalar(
                        "rl_losses/qf2_loss",
                        train_stats["qf2_loss"],
                        self._n_rl_update_steps_total,
                    )
                    self.tb_logger.writer.add_scalar(
                        "rl_losses/policy_loss",
                        train_stats["policy_loss"],
                        self._n_rl_update_steps_total,
                    )
                    self.tb_logger.writer.add_scalar(
                        "rl_losses/alpha_entropy_loss",
                        train_stats["alpha_entropy_loss"],
                        self._n_rl_update_steps_total,
                    )

                    # weights and gradients
                    self.tb_logger.writer.add_scalar(
                        "weights/q1_network",
                        list(self.agent.qf1.parameters())[0].mean(),
                        self._n_rl_update_steps_total,
                    )
                    if list(self.agent.qf1.parameters())[0].grad is not None:
                        param_list = list(self.agent.qf1.parameters())
                        self.tb_logger.writer.add_scalar(
                            "gradients/q1_network",
                            sum(
                                [
                                    param_list[i].grad.mean()
                                    for i in range(len(param_list))
                                ]
                            ),
                            self._n_rl_update_steps_total,
                        )
                    self.tb_logger.writer.add_scalar(
                        "weights/q1_target",
                        list(self.agent.qf1_target.parameters())[0].mean(),
                        self._n_rl_update_steps_total,
                    )
                    if list(self.agent.qf1_target.parameters())[0].grad is not None:
                        param_list = list(self.agent.qf1_target.parameters())
                        self.tb_logger.writer.add_scalar(
                            "gradients/q1_target",
                            sum(
                                [
                                    param_list[i].grad.mean()
                                    for i in range(len(param_list))
                                ]
                            ),
                            self._n_rl_update_steps_total,
                        )
                    self.tb_logger.writer.add_scalar(
                        "weights/q2_network",
                        list(self.agent.qf2.parameters())[0].mean(),
                        self._n_rl_update_steps_total,
                    )
                    if list(self.agent.qf2.parameters())[0].grad is not None:
                        param_list = list(self.agent.qf2.parameters())
                        self.tb_logger.writer.add_scalar(
                            "gradients/q2_network",
                            sum(
                                [
                                    param_list[i].grad.mean()
                                    for i in range(len(param_list))
                                ]
                            ),
                            self._n_rl_update_steps_total,
                        )
                    self.tb_logger.writer.add_scalar(
                        "weights/q2_target",
                        list(self.agent.qf2_target.parameters())[0].mean(),
                        self._n_rl_update_steps_total,
                    )
                    if list(self.agent.qf2_target.parameters())[0].grad is not None:
                        param_list = list(self.agent.qf2_target.parameters())
                        self.tb_logger.writer.add_scalar(
                            "gradients/q2_target",
                            sum(
                                [
                                    param_list[i].grad.mean()
                                    for i in range(len(param_list))
                                ]
                            ),
                            self._n_rl_update_steps_total,
                        )
                    self.tb_logger.writer.add_scalar(
                        "weights/policy",
                        list(self.agent.policy.parameters())[0].mean(),
                        self._n_rl_update_steps_total,
                    )
                    if list(self.agent.policy.parameters())[0].grad is not None:
                        param_list = list(self.agent.policy.parameters())
                        self.tb_logger.writer.add_scalar(
                            "gradients/policy",
                            sum(
                                [
                                    param_list[i].grad.mean()
                                    for i in range(len(param_list))
                                ]
                            ),
                            self._n_rl_update_steps_total,
                        )

            print(
                "Iteration -- {}, Success rate -- {:.3f}, Avg. return -- {:.3f}, Elapsed time {:5d}[s]".format(
                    iteration,
                    np.mean(success_rate),
                    np.mean(np.sum(returns, axis=-1)),
                    int(time.time() - self._start_time),
                )
            )

    def sample_rl_batch(self, tasks, batch_size):
        """sample batch of unordered rl training data from a list/array of tasks"""
        # this batch consists of transitions sampled randomly from replay buffer
        batches = [
            ptu.np_to_pytorch_batch(self.storage.random_batch(task, batch_size))
            for task in tasks
        ]
        unpacked = [utl.unpack_batch(batch) for batch in batches]
        # group elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def _start_training(self):
        self._n_rl_update_steps_total = 0
        self._start_time = time.time()

    def training_mode(self, mode):
        self.agent.train(mode)

    def update_encoding(self, obs, action, reward, done, hidden_state):
        # reset hidden state of the recurrent net when the task is done
        hidden_state = self.vae.encoder.reset_hidden(hidden_state, done)
        with torch.no_grad():  # size should be (batch, dim)
            task_sample, task_mean, task_logvar, hidden_state = self.vae.encoder(
                actions=action.float(),
                states=obs,
                rewards=reward,
                hidden_state=hidden_state,
                return_prior=False,
            )

        return task_sample, task_mean, task_logvar, hidden_state

    @staticmethod
    def get_augmented_obs(obs, mean, logvar):
        mean = mean.reshape((-1, mean.shape[-1]))
        logvar = logvar.reshape((-1, logvar.shape[-1]))
        return torch.cat((obs, mean, logvar), dim=-1)

    def load_model(self, agent_path, device="cpu"):
        self.agent.load_state_dict(torch.load(agent_path, map_location=device))
        self.load_vae()
        self.training_mode(False)
