import os
import time

import gym
import numpy as np
import torch

from algorithms.dqn import DQN, DoubleDQN
from algorithms.sac import SAC
from environments.make_env import make_env
from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from torchkit.networks import FlattenMlp
from data_management.storage_policy import MultiTaskPolicyStorage
from data_management.storage_vae import MultiTaskVAEStorage
from utils import evaluation as utl_eval
from utils.tb_logger import TBLogger
from models.vae import VAE
from models.policy import TanhGaussianPolicy
from utils import logger


class MetaLearner:
    """
    Meta-Learner class.
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

        # initialise environment
        self.env = make_env(
            self.args.env_name,
            self.args.max_rollouts_per_task,
            seed=self.args.seed,
            n_tasks=self.args.num_tasks,
        )

        # unwrapped env to get some info about the environment
        unwrapped_env = self.env.unwrapped
        # split to train/eval tasks
        shuffled_tasks = np.random.permutation(unwrapped_env.get_all_task_idx())
        self.train_tasks = shuffled_tasks[: self.args.num_train_tasks]
        if self.args.num_eval_tasks > 0:
            self.eval_tasks = shuffled_tasks[-self.args.num_eval_tasks :]
        else:
            self.eval_tasks = []
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

        # initialize VAE
        self.vae = VAE(self.args)
        # initialize buffer for VAE updates
        self.vae_storage = MultiTaskVAEStorage(
            max_replay_buffer_size=int(self.args.vae_buffer_size),
            obs_dim=utl.get_dim(self.env.observation_space),
            action_space=self.env.action_space,
            tasks=self.train_tasks,
            trajectory_len=args.max_trajectory_len,
        )

        # initialize policy
        self.initialize_policy()
        # initialize buffer for RL updates
        self.policy_storage = MultiTaskPolicyStorage(
            max_replay_buffer_size=int(self.args.policy_buffer_size),
            obs_dim=self._get_augmented_obs_dim(),
            action_space=self.env.action_space,
            tasks=self.train_tasks,
            trajectory_len=args.max_trajectory_len,
        )

        self.args.belief_reward = False  # initialize arg to not use belief rewards

        logger.log(self.vae.encoder)

    def initialize_policy(self):

        if self.args.policy == "dqn":
            assert (
                self.args.act_space.__class__.__name__ == "Discrete"
            ), "Can't train DQN with continuous action space!"
            q_network = FlattenMlp(
                input_size=self._get_augmented_obs_dim(),
                output_size=self.args.act_space.n,
                hidden_sizes=self.args.dqn_layers,
            )
            self.agent = DQN(
                q_network,
                # optimiser_vae=self.optimizer_vae,
                lr=self.args.policy_lr,
                eps_optim=self.args.dqn_eps,
                alpha_optim=self.args.dqn_alpha,
                gamma=self.args.gamma,
                eps_init=self.args.dqn_epsilon_init,
                eps_final=self.args.dqn_epsilon_final,
                exploration_iters=self.args.dqn_exploration_iters,
                tau=self.args.soft_target_tau,
            ).to(ptu.device)
        elif self.args.policy == "ddqn":
            assert (
                self.args.act_space.__class__.__name__ == "Discrete"
            ), "Can't train DDQN with continuous action space!"
            q_network = FlattenMlp(
                input_size=self._get_augmented_obs_dim(),
                output_size=self.args.act_space.n,
                hidden_sizes=self.args.dqn_layers,
            )
            self.agent = DoubleDQN(
                q_network,
                # optimiser_vae=self.optimizer_vae,
                lr=self.args.policy_lr,
                eps_optim=self.args.dqn_eps,
                alpha_optim=self.args.dqn_alpha,
                gamma=self.args.gamma,
                eps_init=self.args.dqn_epsilon_init,
                eps_final=self.args.dqn_epsilon_final,
                exploration_iters=self.args.dqn_exploration_iters,
                tau=self.args.soft_target_tau,
            ).to(ptu.device)
        elif self.args.policy == "sac":
            assert (
                self.args.act_space.__class__.__name__ == "Box"
            ), "Can't train SAC with discrete action space!"
            q1_network = FlattenMlp(
                input_size=self._get_augmented_obs_dim() + self.args.action_dim,
                output_size=1,
                hidden_sizes=self.args.dqn_layers,
            )
            q2_network = FlattenMlp(
                input_size=self._get_augmented_obs_dim() + self.args.action_dim,
                output_size=1,
                hidden_sizes=self.args.dqn_layers,
            )
            policy = TanhGaussianPolicy(
                obs_dim=self._get_augmented_obs_dim(),
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

        logger.log(self.agent)

    def train(self):
        """
        meta-training loop
        """

        self._start_training()
        for iter_ in range(self.args.num_iters):
            self.training_mode(True)
            # switch to belief reward
            if (
                self.args.switch_to_belief_reward is not None
                and iter_ >= self.args.switch_to_belief_reward
            ):
                self.args.belief_reward = True
            if iter_ == 0:
                print("Collecting initial pool of data..")
                for task in self.train_tasks:
                    self.task_idx = task
                    self.env.reset_task(idx=task)
                    # self.collect_rollouts(num_rollouts=self.args.num_init_rollouts_pool)
                    self.collect_rollouts(
                        num_rollouts=self.args.num_init_rollouts_pool,
                        random_actions=True,
                    )
                print("Done!")
                if self.args.pretrain_len > 0:
                    print("Pre-training for {} updates.".format(self.args.pretrain_len))
                    for update in range(self.args.pretrain_len):
                        indices = np.random.choice(
                            self.train_tasks, self.args.meta_batch
                        )
                        loss, _, _, _, _ = self.update_vae(indices)
                        if (update + 1) % int(self.args.pretrain_len / 10) == 0:
                            print(
                                "Initial VAE training, {} updates. VAE loss: {:.3f}".format(
                                    update + 1, loss.item()
                                )
                            )
                    self._n_vae_update_steps_total += self.args.vae_updates_per_iter

            # collect data from subset of train tasks
            for i in range(self.args.num_tasks_sample):
                task = self.train_tasks[np.random.randint(len(self.train_tasks))]
                self.task_idx = task
                self.env.reset_task(idx=task)
                self.collect_rollouts(num_rollouts=self.args.num_rollouts_per_iter)
            # update
            indices = np.random.choice(self.train_tasks, self.args.meta_batch)
            train_stats = self.update(indices)
            self.training_mode(False)

            if self.args.policy == "dqn":
                self.agent.set_exploration_parameter(iter_ + 1)
            # evaluate and log
            if (iter_ + 1) % self.args.log_interval == 0:
                self.log(iter_ + 1, train_stats)

    def update(self, tasks):
        """
        Meta-update
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

        # --- VAE TRAINING ---
        rew_losses, state_losses, task_losses, kl_terms, vae_losses = [], [], [], [], []
        for update in range(self.args.vae_updates_per_iter):
            # returns mean loss terms
            vae_loss, rew_loss, state_loss, task_loss, kl_term = self.update_vae(tasks)

            rew_losses.append(rew_loss.item())
            state_losses.append(state_loss.item())
            task_losses.append(task_loss.item())
            kl_terms.append(kl_term.item())
            vae_losses.append(vae_loss.item())

        # statistics
        self._n_vae_update_steps_total += self.args.vae_updates_per_iter

        train_stats = {
            **rl_losses_agg,
            **{
                "rew_loss": np.mean(rew_losses),
                "state_loss": np.mean(state_losses),
                "task_loss": np.mean(task_losses),
                "kl_loss": np.mean(kl_terms),
                "vae_loss": np.mean(vae_losses),
            },
        }

        return train_stats

    def evaluate(self, tasks):
        num_episodes = self.args.max_rollouts_per_task
        num_steps_per_episode = self.env.unwrapped._max_episode_steps

        returns_per_episode = np.zeros((len(tasks), num_episodes))
        success_rate = np.zeros(len(tasks))
        task_samples = np.zeros(
            (
                len(tasks),
                self.args.max_trajectory_len + 1,
                self.args.task_embedding_size,
            )
        )
        task_means = np.zeros(
            (
                len(tasks),
                self.args.max_trajectory_len + 1,
                self.args.task_embedding_size,
            )
        )
        task_logvars = np.zeros(
            (
                len(tasks),
                self.args.max_trajectory_len + 1,
                self.args.task_embedding_size,
            )
        )

        if self.args.policy == "dqn":
            reward_preds = np.zeros(
                (len(tasks), self.args.max_trajectory_len + 1, self.env.num_states)
            )
            values = np.zeros((len(tasks), self.args.max_trajectory_len))
        else:
            rewards = np.zeros((len(tasks), self.args.max_trajectory_len))
            reward_preds = np.zeros((len(tasks), self.args.max_trajectory_len))
            obs_size = self.env.unwrapped.observation_space.shape[0]
            observations = np.zeros(
                (len(tasks), self.args.max_trajectory_len + 1, obs_size)
            )
            log_probs = np.zeros((len(tasks), self.args.max_trajectory_len))

        for task_idx, task in enumerate(tasks):
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
            if self.args.fixed_latent_params:
                task_mean = ptu.FloatTensor(
                    utl.vertices(self.args.task_embedding_size)[task]
                ).reshape(task_mean.shape)
                task_logvar = -2.0 * ptu.ones_like(
                    task_logvar
                )  # arbitrary negative enough number
            # store
            task_samples[task_idx, step, :] = ptu.get_numpy(task_sample[0, 0])
            task_means[task_idx, step, :] = ptu.get_numpy(task_mean[0, 0])
            task_logvars[task_idx, step, :] = ptu.get_numpy(task_logvar[0, 0])

            if self.args.policy == "dqn":
                reward_preds[task_idx, step] = ptu.get_numpy(
                    self.vae.reward_decoder(task_sample, None)[0, 0]
                )
            else:
                observations[task_idx, step, :] = ptu.get_numpy(obs[0, :obs_size])

            for episode_idx in range(num_episodes):
                running_reward = 0.0
                for step_idx in range(num_steps_per_episode):
                    # add distribution parameters to observation - policy is conditioned on posterior
                    augmented_obs = self.get_augmented_obs(
                        obs=obs, task_mu=task_mean, task_std=task_logvar
                    )
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
                    if self.args.fixed_latent_params:
                        task_mean = ptu.FloatTensor(
                            utl.vertices(self.args.task_embedding_size)[task]
                        ).reshape(task_mean.shape)
                        task_logvar = -2.0 * ptu.ones_like(
                            task_logvar
                        )  # arbitrary negative enough number
                    # store
                    task_samples[task_idx, step + 1, :] = ptu.get_numpy(
                        task_sample[0, 0]
                    )
                    task_means[task_idx, step + 1, :] = ptu.get_numpy(task_mean[0, 0])
                    task_logvars[task_idx, step + 1, :] = ptu.get_numpy(
                        task_logvar[0, 0]
                    )

                    if self.args.policy == "dqn":
                        reward_preds[task_idx, step + 1, :] = ptu.get_numpy(
                            self.vae.reward_decoder(task_sample, None)[0]
                        )
                        values[task_idx, step] = value.item()
                    elif self.vae.reward_decoder is not None:
                        rewards[task_idx, step] = reward.item()
                        reward_preds[task_idx, step] = ptu.get_numpy(
                            self.vae.reward_decoder(task_sample, next_obs, obs, action)[
                                0, 0
                            ]
                        )
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
            return (
                returns_per_episode,
                success_rate,
                values,
                reward_preds,
                task_samples,
                task_means,
                task_logvars,
            )
        else:
            return (
                returns_per_episode,
                success_rate,
                log_probs,
                observations,
                rewards,
                reward_preds,
                task_samples,
                task_means,
                task_logvars,
            )

    def log(self, iteration, train_stats):
        # --- save models ---
        if iteration % self.args.save_interval == 0:
            save_path = os.path.join(self.tb_logger.full_output_folder, "models")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(
                self.agent.state_dict(),
                os.path.join(save_path, "agent{0}.pt".format(iteration)),
            )
            torch.save(
                self.vae.encoder.state_dict(),
                os.path.join(save_path, "encoder{0}.pt".format(iteration)),
            )
            if self.vae.reward_decoder is not None:
                torch.save(
                    self.vae.reward_decoder.state_dict(),
                    os.path.join(save_path, "reward_decoder{0}.pt".format(iteration)),
                )
            if self.vae.state_decoder is not None:
                torch.save(
                    self.vae.state_decoder.state_dict(),
                    os.path.join(save_path, "state_decoder{0}.pt".format(iteration)),
                )
            if self.vae.task_decoder is not None:
                torch.save(
                    self.vae.task_decoder.state_dict(),
                    os.path.join(save_path, "task_decoder{0}.pt".format(iteration)),
                )

        # evaluate to get more stats
        if self.args.policy == "dqn":
            # get stats on train tasks
            (
                returns_train,
                success_rate_train,
                values,
                reward_preds,
                task_samples,
                task_means,
                task_logvars,
            ) = self.evaluate(self.train_tasks)
        else:
            # get stats on train tasks
            (
                returns_train,
                success_rate_train,
                log_probs,
                observations,
                rewards_train,
                reward_preds_train,
                task_samples,
                task_means,
                task_logvars,
            ) = self.evaluate(self.train_tasks[: len(self.eval_tasks)])
            (
                returns_eval,
                success_rate_eval,
                _,
                observations_eval,
                rewards_eval,
                reward_preds_eval,
                _,
                _,
                _,
            ) = self.evaluate(self.eval_tasks)

        assert self.args.log_tensorboard == True
        if self.args.log_tensorboard:
            # --- log training  ---
            if self.args.policy == "dqn":
                # for i, task in enumerate(self.eval_tasks):
                for i, task in enumerate(self.train_tasks[:5]):
                    self.tb_logger.writer.add_figure(
                        "rewards_pred_task_{}/prior".format(i),
                        utl_eval.vis_rew_pred(
                            self.args, reward_preds[i, 0].round(2), self.env.goals[task]
                        ),
                        self._n_env_steps_total,
                    )
                    self.tb_logger.writer.add_figure(
                        "rewards_pred_task_{}/halfway".format(i),
                        utl_eval.vis_rew_pred(
                            self.args,
                            reward_preds[
                                i, int(np.ceil(reward_preds.shape[1] / 2))
                            ].round(2),
                            self.env.goals[task],
                        ),
                        self._n_env_steps_total,
                    )
                    self.tb_logger.writer.add_figure(
                        "rewards_pred_task_{}/final".format(i),
                        utl_eval.vis_rew_pred(
                            self.args,
                            reward_preds[i, -1].round(2),
                            self.env.goals[task],
                        ),
                        self._n_env_steps_total,
                    )
            else:
                for i, task in enumerate(self.train_tasks[:5]):
                    self.env.reset(task)
                    # self.tb_logger.writer.add_figure('policy_vis_train/task_{}'.format(i),
                    #                                  utl_eval.plot_rollouts(observations[i, :], self.env),
                    #                                  self._n_env_steps_total)
                    # # sample batch
                    # obs, _, _, _, _ = self.sample_rl_batch(tasks=[task],
                    #                                        batch_size=self.policy_storage.task_buffers[task].size())
                    # self.tb_logger.writer.add_figure('state_space_coverage/task_{}'.format(i),
                    #                                  utl_eval.plot_visited_states(ptu.get_numpy(obs[0]), self.env),
                    #                                  self._n_env_steps_total)
                    self.tb_logger.writer.add_figure(
                        "reward_prediction_train/task_{}".format(i),
                        utl_eval.plot_rew_pred_vs_rew(
                            rewards_train[i, :], reward_preds_train[i, :]
                        ),
                        self._n_env_steps_total,
                    )
                for i, task in enumerate(self.eval_tasks[:5]):
                    self.env.reset(task)
                    # self.tb_logger.writer.add_figure('policy_vis_eval/task_{}'.format(i),
                    #                                  utl_eval.plot_rollouts(observations_eval[i, :], self.env),
                    #                                  self._n_env_steps_total)
                    self.tb_logger.writer.add_figure(
                        "reward_prediction_eval/task_{}".format(i),
                        utl_eval.plot_rew_pred_vs_rew(
                            rewards_eval[i, :], reward_preds_eval[i, :]
                        ),
                        self._n_env_steps_total,
                    )
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
                if self.args.policy != "dqn":
                    self.tb_logger.writer.add_scalar(
                        "returns_multi_episode/sum_eval",
                        np.mean(np.sum(returns_eval, axis=-1)),
                        self._n_env_steps_total,
                    )
                    self.tb_logger.writer.add_scalar(
                        "returns_multi_episode/success_rate_eval",
                        np.mean(success_rate_eval),
                        self._n_env_steps_total,
                    )
            else:
                # self.tb_logger.writer.add_scalar('returns/returns_mean', np.mean(returns),
                #                                  self._n_env_steps_total)
                # self.tb_logger.writer.add_scalar('returns/returns_std', np.std(returns),
                #                                  self._n_env_steps_total)
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
                # self.tb_logger.writer.add_scalar('returns/success_rate', np.mean(success_rate),
                #                                  self._n_env_steps_total)
                self.tb_logger.writer.add_scalar(
                    "returns/success_rate_train",
                    np.mean(success_rate_train),
                    self._n_env_steps_total,
                )
            # encoder
            self.tb_logger.writer.add_scalar(
                "encoder/task_embedding_init",
                task_samples[:, 0].mean(),
                self._n_env_steps_total,
            )
            self.tb_logger.writer.add_scalar(
                "encoder/task_mu_init", task_means[:, 0].mean(), self._n_env_steps_total
            )
            self.tb_logger.writer.add_scalar(
                "encoder/task_logvar_init",
                task_logvars[:, 0].mean(),
                self._n_env_steps_total,
            )
            self.tb_logger.writer.add_scalar(
                "encoder/task_embedding_halfway",
                task_samples[:, int(task_samples.shape[-1] / 2)].mean(),
                self._n_env_steps_total,
            )
            self.tb_logger.writer.add_scalar(
                "encoder/task_mu_halfway",
                task_means[:, int(task_means.shape[-1] / 2)].mean(),
                self._n_env_steps_total,
            )
            self.tb_logger.writer.add_scalar(
                "encoder/task_logvar_halfway",
                task_logvars[:, int(task_logvars.shape[-1] / 2)].mean(),
                self._n_env_steps_total,
            )
            self.tb_logger.writer.add_scalar(
                "encoder/task_embedding_final",
                task_samples[:, -1].mean(),
                self._n_env_steps_total,
            )
            self.tb_logger.writer.add_scalar(
                "encoder/task_mu_final",
                task_means[:, -1].mean(),
                self._n_env_steps_total,
            )
            self.tb_logger.writer.add_scalar(
                "encoder/task_logvar_final",
                task_logvars[:, -1].mean(),
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
                    "rl_losses/alpha_loss",
                    train_stats["alpha_loss"],
                    self._n_env_steps_total,
                )

            # VAE losses
            self.tb_logger.writer.add_scalar(
                "vae_losses/vae_loss", train_stats["vae_loss"], self._n_env_steps_total
            )
            self.tb_logger.writer.add_scalar(
                "vae_losses/kl_loss", train_stats["kl_loss"], self._n_env_steps_total
            )
            if self.vae.reward_decoder is not None:
                self.tb_logger.writer.add_scalar(
                    "vae_losses/reward_rec_loss",
                    train_stats["rew_loss"],
                    self._n_env_steps_total,
                )
            if self.vae.state_decoder is not None:
                self.tb_logger.writer.add_scalar(
                    "vae_losses/state_rec_loss",
                    train_stats["state_loss"],
                    self._n_env_steps_total,
                )
            if self.vae.task_decoder is not None:
                self.tb_logger.writer.add_scalar(
                    "vae_losses/task_rec_loss",
                    train_stats["task_loss"],
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

            self.tb_logger.writer.add_scalar(
                "weights/encoder",
                list(self.vae.encoder.parameters())[0].mean(),
                self._n_env_steps_total,
            )
            if list(self.vae.encoder.parameters())[0].grad is not None:
                param_list = list(self.vae.encoder.parameters())
                self.tb_logger.writer.add_scalar(
                    "gradients/encoder",
                    sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                    self._n_env_steps_total,
                )

            if self.vae.reward_decoder is not None:
                self.tb_logger.writer.add_scalar(
                    "weights/reward_decoder",
                    list(self.vae.reward_decoder.parameters())[0].mean(),
                    self._n_env_steps_total,
                )
                if list(self.vae.reward_decoder.parameters())[0].grad is not None:
                    param_list = list(self.vae.reward_decoder.parameters())
                    self.tb_logger.writer.add_scalar(
                        "gradients/reward_decoder",
                        sum(
                            [param_list[i].grad.mean() for i in range(len(param_list))]
                        ),
                        self._n_env_steps_total,
                    )
            if self.vae.state_decoder is not None:
                self.tb_logger.writer.add_scalar(
                    "weights/state_decoder",
                    list(self.vae.state_decoder.parameters())[0].mean(),
                    self._n_env_steps_total,
                )
                if list(self.vae.state_decoder.parameters())[0].grad is not None:
                    param_list = list(self.vae.state_decoder.parameters())
                    self.tb_logger.writer.add_scalar(
                        "gradients/state_decoder",
                        sum(
                            [param_list[i].grad.mean() for i in range(len(param_list))]
                        ),
                        self._n_env_steps_total,
                    )
            if self.vae.task_decoder is not None:
                self.tb_logger.writer.add_scalar(
                    "weights/task_decoder",
                    list(self.vae.task_decoder.parameters())[0].mean(),
                    self._n_env_steps_total,
                )
                if list(self.vae.task_decoder.parameters())[0].grad is not None:
                    param_list = list(self.vae.task_decoder.parameters())
                    self.tb_logger.writer.add_scalar(
                        "gradients/task_decoder",
                        sum(
                            [param_list[i].grad.mean() for i in range(len(param_list))]
                        ),
                        self._n_env_steps_total,
                    )

        # output to user
        # print("Iteration -- {:3d}, Num. RL updates -- {:6d}, Elapsed time {:5d}[s]".
        #       format(iteration,
        #              self._n_rl_update_steps_total,
        #              int(time.time() - self._start_time)))
        print(
            "Iteration -- {}, Success rate train -- {:.3f}, Success rate eval.-- {:.3f}, "
            "Avg. return train -- {:.3f}, Avg. return eval. -- {:.3f}, Elapsed time {:5d}[s]".format(
                iteration,
                np.mean(success_rate_train),
                np.mean(success_rate_eval),
                np.mean(np.sum(returns_train, axis=-1)),
                np.mean(np.sum(returns_eval, axis=-1)),
                int(time.time() - self._start_time),
            )
        )
        # Our logger
        logger.record_step(self._n_env_steps_total)
        logger.record_tabular("z/env_steps", self._n_env_steps_total)
        logger.record_tabular("z/time_cost", int(time.time() - self._start_time))
        logger.record_tabular("z/rl_steps", self._n_rl_update_steps_total)
        logger.record_tabular("z/vae_steps", self._n_vae_update_steps_total)
        logger.record_tabular("z/rollouts", self._n_rollouts_total)

        for episode_idx in range(self.args.max_rollouts_per_task):
            logger.record_tabular(
                "metrics/return_train_episode_{}".format(episode_idx + 1),
                np.mean(returns_train[:, episode_idx]),
            )
            logger.record_tabular(
                "metrics/return_eval_episode_{}".format(episode_idx + 1),
                np.mean(returns_eval[:, episode_idx]),
            )
        logger.record_tabular(
            "metrics/return_train_total", np.mean(np.sum(returns_train, axis=-1))
        )
        logger.record_tabular(
            "metrics/return_eval_total", np.mean(np.sum(returns_eval, axis=-1))
        )

        logger.dump_tabular()

    def update_vae(self, tasks):
        """
        Compute losses, update parameters and return the VAE losses
        """

        # get a mini-batch of episodes
        obs, actions, rewards, next_obs, terms = self.sample_vae_batch(
            tasks, self.args.vae_batch_num_rollouts_per_task
        )

        episode_len, num_episodes, _ = obs.shape

        # get time-steps for ELBO computation
        if self.args.vae_batch_num_elbo_terms is not None:
            elbo_timesteps = np.stack(
                [
                    np.random.choice(
                        range(0, self.vae_storage.trajectory_len + 1),
                        self.args.vae_batch_num_elbo_terms,
                        replace=False,
                    )
                    for _ in range(num_episodes)
                ]
            )
        else:
            elbo_timesteps = np.repeat(
                np.arange(0, self.vae_storage.trajectory_len + 1).reshape(1, -1),
                num_episodes,
                axis=0,
            )

        # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
        _, latent_mean, latent_logvar, _ = self.vae.encoder(
            actions=actions,
            states=next_obs,
            rewards=rewards,
            hidden_state=None,
            return_prior=True,
        )

        rew_recon_losses, state_recon_losses, task_recon_losses, kl_terms = (
            [],
            [],
            [],
            [],
        )

        # for each task we have in our batch
        for episode_idx in range(num_episodes):

            # get the embedding values (size: traj_length+1 * latent_dim; the +1 is for the prior)
            curr_means = latent_mean[: episode_len + 1, episode_idx, :]
            curr_logvars = latent_logvar[: episode_len + 1, episode_idx, :]
            # take one sample for each ELBO term
            curr_samples = self.vae.encoder._sample_gaussian(curr_means, curr_logvars)

            # select data from current rollout (result is traj_length * obs_dim)
            curr_obs = obs[:, episode_idx, :]
            curr_next_obs = next_obs[:, episode_idx, :]
            curr_actions = actions[:, episode_idx, :]
            curr_rewards = rewards[:, episode_idx, :]

            num_latents = curr_samples.shape[0]  # includes the prior
            num_decodes = curr_obs.shape[0]

            # expand the latent to match the (x, y) pairs of the decoder
            dec_embedding = (
                curr_samples.unsqueeze(0)
                .expand((num_decodes, *curr_samples.shape))
                .transpose(1, 0)
            )
            dec_embedding_task = curr_samples

            # expand the (x, y) pair of the encoder
            dec_obs = curr_obs.unsqueeze(0).expand((num_latents, *curr_obs.shape))
            dec_next_obs = curr_next_obs.unsqueeze(0).expand(
                (num_latents, *curr_next_obs.shape)
            )
            dec_actions = curr_actions.unsqueeze(0).expand(
                (num_latents, *curr_actions.shape)
            )
            dec_rewards = curr_rewards.unsqueeze(0).expand(
                (num_latents, *curr_rewards.shape)
            )

            if self.args.decode_reward:
                # compute reconstruction loss for this trajectory
                # (for each timestep that was encoded, decode everything and sum it up)
                rrl = self.vae.compute_rew_reconstruction_loss(
                    dec_embedding, dec_obs, dec_next_obs, dec_actions, dec_rewards
                )
                # sum along the trajectory which we decoded (sum in ELBO_t)
                if self.args.decode_only_past:
                    curr_idx = 0
                    past_reconstr_sum = []
                    for i, idx_timestep in enumerate(elbo_timesteps[episode_idx]):
                        dec_until = idx_timestep
                        if dec_until != 0:
                            past_reconstr_sum.append(
                                rrl[curr_idx : curr_idx + dec_until].sum()
                            )
                        curr_idx += dec_until
                    rrl = torch.stack(past_reconstr_sum)
                else:
                    rrl = rrl.sum(dim=1)
                rew_recon_losses.append(rrl)
            if self.args.decode_state:
                srl = self.vae.compute_state_reconstruction_loss(
                    dec_embedding, dec_obs, dec_next_obs, dec_actions
                )
                srl = srl.sum(dim=1)
                state_recon_losses.append(srl)
            if self.args.decode_task:
                trl = self.vae.compute_task_reconstruction_loss(
                    dec_embedding_task, tasks[episode_idx]
                )
                task_recon_losses.append(trl)
            if not self.args.disable_stochasticity_in_latent:
                # compute the KL term for each ELBO term of the current trajectory
                kl = self.vae.compute_kl_loss(
                    curr_means, curr_logvars, elbo_timesteps[episode_idx]
                )
                kl_terms.append(kl)

        # sum the ELBO_t terms per task
        if self.args.decode_reward:
            rew_recon_losses = torch.stack(rew_recon_losses)
            rew_recon_losses = rew_recon_losses.sum(dim=1)
        else:
            rew_recon_losses = ptu.zeros(1)  # 0 -- but with option of .mean()

        if self.args.decode_state:
            state_recon_losses = torch.stack(state_recon_losses)
            state_recon_losses = state_recon_losses.sum(dim=1)
        else:
            state_recon_losses = ptu.zeros(1)

        if self.args.decode_task:
            task_recon_losses = torch.stack(task_recon_losses)
            task_recon_losses = task_recon_losses.sum(dim=1)
        else:
            task_recon_losses = ptu.zeros(1)

        if not self.args.disable_stochasticity_in_latent:
            kl_terms = torch.stack(kl_terms)
            kl_terms = kl_terms.sum(dim=1)
        else:
            kl_terms = ptu.zeros(1)

        # take average (this is the expectation over p(M))
        loss = (
            self.args.rew_loss_coeff * rew_recon_losses
            + self.args.state_loss_coeff * state_recon_losses
            + self.args.task_loss_coeff * task_recon_losses
            + self.args.kl_weight * kl_terms
        ).mean()

        # make sure we can compute gradients
        if not self.args.disable_stochasticity_in_latent:
            assert kl_terms.requires_grad
        if self.args.decode_reward:
            assert rew_recon_losses.requires_grad
        if self.args.decode_state:
            assert state_recon_losses.requires_grad
        if self.args.decode_task:
            assert task_recon_losses.requires_grad

        # update
        self.vae.optimizer.zero_grad()
        loss.backward()
        self.vae.optimizer.step()

        return (
            loss,
            rew_recon_losses.mean(),
            state_recon_losses.mean(),
            task_recon_losses.mean(),
            kl_terms.mean(),
        )

    def training_mode(self, mode):
        # policy
        self.agent.train(mode)
        # encoder
        self.vae.encoder.train(mode)
        # decoders
        if self.args.decode_reward:
            self.vae.reward_decoder.train(mode)
        if self.args.decode_state:
            self.vae.state_decoder.train(mode)
        if self.args.decode_task:
            self.vae.task_decoder.train(mode)

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
            # reset episode (length)
            self.vae_storage.reset_running_episode(self.task_idx)
            # self.policy_storage.reset_running_episode(self.task_idx)

            # get prior parameters
            with torch.no_grad():
                _, task_mean, task_logvar, hidden_state = self.encode_running_episode()
            # if self.args.fixed_latent_params:
            #     assert 2 ** self.args.task_embedding_size >= self.args.num_tasks
            #     task_mean = ptu.FloatTensor(utl.vertices(self.args.task_embedding_size)[self.task_idx])
            #     task_logvar = -2. * ptu.ones_like(task_logvar)   # arbitrary negative enough number
            # add distribution parameters to observation - policy is conditioned on posterior
            augmented_obs = self.get_augmented_obs(
                obs=obs, task_mu=task_mean, task_std=task_logvar
            )

            while not done_rollout:
                if random_actions:
                    if self.args.policy == "dqn":
                        action = ptu.FloatTensor(
                            [[self.env.action_space.sample()]]
                        ).type(
                            torch.long
                        )  # Sample random action
                    else:
                        action = ptu.FloatTensor([self.env.action_space.sample()])
                else:
                    if self.args.policy == "dqn":
                        action, _ = self.agent.act(obs=augmented_obs)  # DQN
                    else:
                        action, _, _, _ = self.agent.act(obs=augmented_obs)  # SAC
                # observe reward and next obs
                next_obs, reward, done, info = utl.env_step(
                    self.env, action.squeeze(dim=0)
                )
                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                # belief reward - averaging over multiple latent embeddings - R+ = E[R(b)]
                if self.args.belief_reward:
                    if self.args.policy == "dqn" and self.args.oracle_belief_rewards:
                        belief_reward = np.array([info["belief_reward"]])
                    else:
                        belief_reward = self.vae.compute_belief_reward(
                            task_mean,
                            task_logvar,
                            obs=obs,
                            next_obs=next_obs,
                            actions=action,
                        ).view(-1, 1)
                        belief_reward = ptu.get_numpy(belief_reward.squeeze(dim=0))

                # update encoding
                _, task_mean, task_logvar, hidden_state = self.update_encoding(
                    obs=next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    hidden_state=hidden_state,
                )
                if self.args.fixed_latent_params:
                    task_mean = ptu.FloatTensor(
                        utl.vertices(self.args.task_embedding_size)[self.task_idx]
                    ).reshape(task_mean.shape)
                    task_logvar = -2.0 * ptu.ones_like(
                        task_logvar
                    )  # arbitrary negative enough number
                # get augmented next obs
                augmented_next_obs = self.get_augmented_obs(
                    obs=next_obs, task_mu=task_mean, task_std=task_logvar
                )

                # add data to vae buffer - (s, a, r, s', term)
                self.vae_storage.add_sample(
                    task=self.task_idx,
                    observation=ptu.get_numpy(obs.squeeze(dim=0)),
                    action=ptu.get_numpy(action.squeeze(dim=0)),
                    reward=ptu.get_numpy(reward.squeeze(dim=0)),
                    terminal=ptu.get_numpy(done.squeeze(dim=0)),
                    next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                )

                # add data to policy buffer - (s+, a, r, s'+, term)
                term = (
                    self.env.unwrapped.is_goal_state()
                    if "is_goal_state" in dir(self.env.unwrapped)
                    else False
                )
                self.policy_storage.add_sample(
                    task=self.task_idx,
                    observation=ptu.get_numpy(augmented_obs.squeeze(dim=0)),
                    action=ptu.get_numpy(action.squeeze(dim=0)),
                    reward=belief_reward
                    if self.args.belief_reward
                    else ptu.get_numpy(reward.squeeze(dim=0)),
                    terminal=np.array([term], dtype=float),
                    next_observation=ptu.get_numpy(augmented_next_obs.squeeze(dim=0)),
                )

                # set: obs <- next_obs
                obs = next_obs.clone()
                augmented_obs = augmented_next_obs.clone()

                # update statistics
                self._n_env_steps_total += 1
                if (
                    "is_goal_state" in dir(self.env.unwrapped)
                    and self.env.unwrapped.is_goal_state()
                ):  # count successes
                    self._successes_in_buffer += 1
            self._n_rollouts_total += 1

    def encode_running_episode(self):
        """
        (Re-)Encodes (for each process) the entire current trajectory.
        Returns sample/mean/logvar and hidden state (if applicable) for the current timestep.
        :param reset_task:
        :return:
        """

        # get the current batch (zero-padded obs/act/rew + length indicators)
        obs, next_obs, act, rew, length = self.vae_storage.get_running_episode(
            task=self.task_idx
        )
        # convert numpy arrays to torch tensors
        obs, next_obs, act, rew = ptu.list_from_numpy([obs, next_obs, act, rew])
        # get embedding - will return (1+sequence_len) * batch * input_size -- includes the prior!
        (
            all_task_samples,
            all_task_means,
            all_task_logvars,
            all_hidden_states,
        ) = self.vae.encoder(
            actions=act,
            states=next_obs,
            rewards=rew,
            hidden_state=None,
            return_prior=True,
        )

        # get the embedding / hidden state of the current time step (need to do this since we zero-padded)
        posterior_sample = all_task_samples[length][0].detach().to(ptu.device)
        task_mean = all_task_means[length][0].detach().to(ptu.device)
        task_logvar = all_task_logvars[length][0].detach().to(ptu.device)
        if self.args.encoder_type == "rnn":
            hidden_state = all_hidden_states[length][0].detach().to(ptu.device)
        else:
            raise NotImplementedError

        return posterior_sample, task_mean, task_logvar, hidden_state

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

    def get_augmented_obs(self, obs, task_sample=None, task_mu=None, task_std=None):

        augmented_obs = obs.clone()

        if self.args.sample_embeddings and (task_sample is not None):
            augmented_obs = torch.cat((augmented_obs, task_sample), dim=1)
        elif (task_mu is not None) and (task_std is not None):
            task_mu = task_mu.reshape((-1, task_mu.shape[-1]))
            task_std = task_std.reshape((-1, task_std.shape[-1]))
            augmented_obs = torch.cat((augmented_obs, task_mu, task_std), dim=-1)

        return augmented_obs

    def _get_augmented_obs_dim(self):
        dim = utl.get_dim(self.env.observation_space)
        if self.args.sample_embeddings:
            dim += self.args.task_embedding_size
        else:
            dim += 2 * self.args.task_embedding_size

        return dim

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

    def sample_vae_batch(self, tasks, rollouts_per_task=1):
        """sample batch of episodes for vae training from a list/array of tasks"""

        batches = [
            ptu.np_to_pytorch_batch(
                self.vae_storage.random_episodes(task, rollouts_per_task)
            )
            for task in tasks
        ]
        unpacked = [utl.unpack_batch(batch) for batch in batches]
        # group elements together
        unpacked = [
            [
                x[i]
                .reshape(rollouts_per_task, -1, x[i].shape[-1])
                .transpose(0, 1)
                .unsqueeze(dim=0)
                for x in unpacked
            ]
            for i in range(len(unpacked[0]))
        ]
        batch = []
        for x in unpacked:
            x = torch.cat(x, dim=0).transpose(
                0, 1
            )  # dims: (traj_len, n_tasks, rollouts_per_task, dim)
            # x = torch.cat(x, dim=0)
            # flatten out task dim
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            # append to output batch
            batch.append(x)

        return batch

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
        if "encoder_path" in kwargs:
            self.vae.encoder.load_state_dict(
                torch.load(kwargs["encoder_path"], map_location=device)
            )
        if "reward_decoder_path" in kwargs and self.vae.reward_decoder is not None:
            self.vae.reward_decoder.load_state_dict(
                torch.load(kwargs["reward_decoder_path"], map_location=device)
            )
        if "state_decoder_path" in kwargs and self.vae.state_decoder is not None:
            self.vae.state_decoder.load_state_dict(
                torch.load(kwargs["state_decoder_path"], map_location=device)
            )
        if "task_decoder_path" in kwargs and self.vae.task_decoder is not None:
            self.vae.task_decoder.load_state_dict(
                torch.load(kwargs["task_decoder_path"], map_location=device)
            )

        self.training_mode(False)
