import os
import time
import inspect

import numpy as np
import torch

import gym
from envs.meta.make_env import make_env
from policies.models.policy_mlp import ModelFreeOffPolicy_MLP as Policy_MLP
from buffers.simple_replay_buffer import SimpleReplayBuffer

from BOReL.models.vae import VAE

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from utils import evaluation as utl_eval
from utils import logger


class MetaLearner:
    """
    Meta-Learner class.
    """

    def __init__(self, env_args, train_args, policy_args, vae_args, seed, **kwargs):
        """
        Initialises: environments, policy (+storage +optimiser).
        """
        self.seed = seed

        self.init_env(**env_args)

        self.init_vae(**vae_args)

        self.init_policy(**policy_args)

        self.init_train(**train_args)

    def init_env(
        self,
        env_name,
        max_rollouts_per_task,
        num_tasks,
        num_train_tasks,
        num_eval_tasks,
        **kwargs
    ):
        # initialise environment, using varibad wrapper
        if num_tasks > 0:  # meta tasks
            self.env = make_env(
                env_name,
                max_rollouts_per_task,
                seed=self.seed,
                n_tasks=num_tasks,
                **kwargs
            )

            # unwrapped env to get some info about the environment
            unwrapped_env = self.env.unwrapped

            # split to train/eval tasks
            assert num_train_tasks >= num_eval_tasks
            shuffled_tasks = np.random.permutation(unwrapped_env.get_all_task_idx())
            self.train_tasks = shuffled_tasks[:num_train_tasks]
            self.eval_tasks = shuffled_tasks[-num_eval_tasks:]

            # calculate what the maximum length of the trajectories is
            self.max_rollouts_per_task = max_rollouts_per_task
            self.max_trajectory_len = self.env.horizon_bamdp  # H^+ = N * H

        else:  # pomdp task, not using varibad wrapper
            assert num_tasks == num_train_tasks == 0
            self.env = gym.make(env_name)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.env._max_episode_steps

        # get action / observation dimensions
        assert self.env.action_space.__class__.__name__ == "Box"
        self.act_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]  # include 1-dim done
        logger.log(
            "obs_dim",
            self.obs_dim,
            "act_dim",
            self.act_dim,
            "max trajectory len",
            self.max_trajectory_len,
        )
        logger.log(ptu.device)

    def init_vae(
        self, task_embedding_size, encoder, decoder, optim, buffer_size, **kwargs
    ):
        # initialize VAE
        self.vae = VAE(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            task_embedding_size=task_embedding_size,
            encoder=encoder,
            decoder=decoder,
            optim=optim,
        )

        # initialize buffer for VAE updates
        # actually this can be merged into policy buffer but requires
        # (1) ignore belief (2) done=True (3) different size
        self.vae_storage = SimpleReplayBuffer(
            max_replay_buffer_size=int(buffer_size),
            observation_dim=self.obs_dim,
            action_dim=self.act_dim,
            max_trajectory_len=self.max_trajectory_len,
        )

    def init_policy(
        self,
        policy: str = "sac",
        buffer_size=1e6,
        sample_embeddings: bool = False,
        switch_to_belief_reward=None,
        num_belief_samples=None,
        **kwargs
    ):
        # initialize policy, NOTE: now assume SAC
        assert policy == "sac"
        self.policy_type = policy
        assert sample_embeddings == False  # not posterior sampling
        self.sample_embeddings = sample_embeddings

        self.agent = Policy_MLP(
            algo=self.policy_type,
            obs_dim=self._get_augmented_obs_dim(),
            action_dim=self.act_dim,
            **kwargs
        ).to(ptu.device)
        logger.log(self.agent)
        logger.log("augmented obs dim (obs+belief)", self._get_augmented_obs_dim())

        # initialize buffer for RL updates
        self.policy_storage = SimpleReplayBuffer(
            max_replay_buffer_size=int(buffer_size),
            observation_dim=self._get_augmented_obs_dim(),
            action_dim=self.act_dim,
            max_trajectory_len=self.max_trajectory_len,
        )

        self.belief_reward = False  # initialize arg to not use belief rewards
        self.num_belief_samples = num_belief_samples  # 40
        self.switch_to_belief_reward = switch_to_belief_reward
        assert switch_to_belief_reward is None

    def init_train(
        self,
        num_iters,
        num_init_rollouts_pool,
        num_rollouts_per_iter,
        rl_updates_per_iter,
        policy_batch_size,
        vae_updates_per_iter,
        vae_batch_num_rollouts,
        log_interval,
        save_interval,
        log_tensorboard,
        **kwargs
    ):
        self.num_iters = num_iters
        self.num_init_rollouts_pool = num_init_rollouts_pool
        self.num_rollouts_per_iter = num_rollouts_per_iter

        total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
        logger.log(
            "*** total rollouts",
            total_rollouts,
            "total env steps",
            self.max_trajectory_len * total_rollouts,
        )

        self.rl_updates_per_iter = rl_updates_per_iter
        self.vae_updates_per_iter = vae_updates_per_iter
        self.policy_batch_size = policy_batch_size
        self.vae_batch_num_rollouts = vae_batch_num_rollouts

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.log_tensorboard = log_tensorboard

    def train(self):
        """
        meta-training loop
        NOTE: the main difference from BORel to varibad is changing the alternation
        of rollout collection and model updates: in varibad, one step collection is
        followed by several model updates; while in borel, we collect several **entire**
        trajectories from random sampled tasks for each model updates.
        """

        self._start_training()
        for iter_ in range(self.num_iters):
            # print(iter_)
            self.training_mode(True)
            # switch to belief reward
            if (
                self.switch_to_belief_reward is not None
                and iter_ >= self.switch_to_belief_reward
            ):
                self.belief_reward = True

            if iter_ == 0:
                logger.log("Collecting initial pool of data..")
                self.collect_rollouts(
                    num_rollouts=self.num_init_rollouts_pool, random_actions=True
                )
                logger.log("Done!")
                # NOTE: varibad does not need pre-training vae

            # collect data from subset of train tasks: num_tasks_sample*num_rollouts_per_iter
            self.collect_rollouts(num_rollouts=self.num_rollouts_per_iter)

            # update
            train_stats = self.update(
                self.num_init_rollouts_pool
                if iter_ == 0 and len(self.train_tasks) == 0
                else 1
            )
            self.training_mode(False)
            self.log_train_stats(train_stats)

            # evaluate and log
            if (iter_ + 1) % self.log_interval == 0:
                self.log(iter_ + 1)

    def update(self, multiplier: int):
        """
        Meta-update
        """

        # --- RL TRAINING ---
        rl_losses_agg = {}
        for update in range(multiplier * self.rl_updates_per_iter):
            # sample random RL batch: in transitions
            batch = self.sample_rl_batch(self.policy_batch_size)

            # RL update
            rl_losses = self.agent.update(batch)

            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg["rl_loss/" + k] = [v]
                else:  # append values
                    rl_losses_agg["rl_loss/" + k].append(v)
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += self.rl_updates_per_iter

        # --- VAE TRAINING ---
        vae_losses_agg = {}
        for update in range(multiplier * self.vae_updates_per_iter):
            # sample random vae batch: in trajectories
            # print("vae mem", torch.cuda.memory_allocated(0))
            batch = self.sample_vae_batch(self.vae_batch_num_rollouts)

            # vae update
            vae_losses = self.vae.update(
                batch["obs"], batch["act"], batch["rew"], batch["obs2"]
            )

            for k, v in vae_losses.items():
                if update == 0:  # first iterate - create list
                    vae_losses_agg["vae_loss/" + k] = [v]
                else:  # append values
                    vae_losses_agg["vae_loss/" + k].append(v)
        # statistics
        for k in vae_losses_agg:
            vae_losses_agg[k] = np.mean(vae_losses_agg[k])
        self._n_vae_update_steps_total += self.vae_updates_per_iter

        return {**rl_losses_agg, **vae_losses_agg}

    @torch.no_grad()
    def evaluate(self, tasks):
        num_episodes = self.max_rollouts_per_task  # k
        # max_trajectory_len = k*H
        returns_per_episode = np.zeros((len(tasks), num_episodes))
        success_rate = np.zeros(len(tasks))

        if len(self.train_tasks) > 0:
            num_steps_per_episode = self.env.unwrapped._max_episode_steps  # H
            obs_size = self.env.unwrapped.observation_space.shape[0]  # original size
            observations = np.zeros((len(tasks), self.max_trajectory_len + 1, obs_size))
        else:
            num_steps_per_episode = self.env._max_episode_steps
            observations = None

        task_samples = np.zeros(
            (len(tasks), self.max_trajectory_len + 1, self.vae.task_embedding_size)
        )
        task_means = np.zeros(
            (len(tasks), self.max_trajectory_len + 1, self.vae.task_embedding_size)
        )
        task_logvars = np.zeros(
            (len(tasks), self.max_trajectory_len + 1, self.vae.task_embedding_size)
        )

        rewards = np.zeros((len(tasks), self.max_trajectory_len))
        reward_preds = np.zeros((len(tasks), self.max_trajectory_len))

        for task_idx, task in enumerate(tasks):
            if "task" in inspect.getargspec(self.env.reset)[0]:
                obs = ptu.from_numpy(self.env.reset(task=task))  # reset meta task
            else:
                obs = ptu.from_numpy(self.env.reset())  # reset pomdp/mdp

            obs = obs.reshape(-1, obs.shape[-1])
            step = 0

            # get prior parameters (1,1,embed_size)
            task_sample, task_mean, task_logvar, hidden_state = self.reset_encoding()

            # store
            task_samples[task_idx, step, :] = ptu.get_numpy(task_sample[0, 0])
            task_means[task_idx, step, :] = ptu.get_numpy(task_mean[0, 0])
            task_logvars[task_idx, step, :] = ptu.get_numpy(task_logvar[0, 0])

            if len(self.train_tasks) > 0:
                observations[task_idx, step, :] = ptu.get_numpy(obs[0, :obs_size])

            for episode_idx in range(num_episodes):
                running_reward = 0.0
                for _ in range(num_steps_per_episode):
                    # ignore done=True, proceed till horizon H, but actually fixed horizon...
                    # add distribution parameters to observation - policy is conditioned on posterior
                    augmented_obs = self.get_augmented_obs(
                        obs=obs, task_mu=task_mean, task_std=task_logvar
                    )
                    action, _, _, _ = self.agent.act(
                        obs=augmented_obs, deterministic=True
                    )

                    # observe reward and next obs
                    next_obs, reward, done, _ = utl.env_step(
                        self.env, action.squeeze(dim=0)
                    )
                    running_reward += reward.item()
                    # update encoding (1,embed_size)
                    (
                        task_sample,
                        task_mean,
                        task_logvar,
                        hidden_state,
                    ) = self.update_encoding(
                        obs=next_obs,
                        action=action,
                        reward=reward,
                        hidden_state=hidden_state,
                    )

                    # store
                    task_samples[task_idx, step + 1, :] = ptu.get_numpy(task_sample[0])
                    task_means[task_idx, step + 1, :] = ptu.get_numpy(task_mean[0])
                    task_logvars[task_idx, step + 1, :] = ptu.get_numpy(task_logvar[0])

                    rewards[task_idx, step] = reward.item()
                    if self.vae.reward_decoder is not None:
                        reward_preds[task_idx, step] = self.vae.reward_decoder(
                            task_sample, next_obs, obs, action
                        ).item()
                    if len(self.train_tasks) > 0:
                        observations[task_idx, step + 1, :] = ptu.get_numpy(
                            next_obs[0, :obs_size]
                        )

                    if (
                        "is_goal_state" in dir(self.env.unwrapped)
                        and self.env.unwrapped.is_goal_state()
                    ):
                        success_rate[task_idx] = 1.0  # ever once reach
                    # set: obs <- next_obs
                    obs = next_obs.clone()
                    step += 1

                returns_per_episode[task_idx, episode_idx] = running_reward

        return (
            returns_per_episode,
            success_rate,
            observations,
            rewards,
            reward_preds,
            task_samples,
            task_means,
            task_logvars,
        )

    def log_train_stats(self, train_stats):
        logger.record_step(self._n_env_steps_total)
        ## log losses
        for k, v in train_stats.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()

    def log(self, iteration):
        # --- save models ---
        # if iteration % self.save_interval == 0:
        # 	save_path = os.path.join(logger.get_dir(), 'save')
        # 	torch.save(self.agent.state_dict(), os.path.join(save_path, "agent{0}.pt".format(iteration)))
        # 	torch.save(self.vae.encoder.state_dict(), os.path.join(save_path, "encoder{0}.pt".format(iteration)))
        # 	if self.vae.reward_decoder is not None:
        # 		torch.save(self.vae.reward_decoder.state_dict(), os.path.join(save_path, "reward_decoder{0}.pt".format(iteration)))
        # 	if self.vae.state_decoder is not None:
        # 		torch.save(self.vae.state_decoder.state_dict(), os.path.join(save_path, "state_decoder{0}.pt".format(iteration)))

        # --- evaluation ----
        if len(self.train_tasks) > 0:
            (
                returns_train,
                success_rate_train,
                observations,
                rewards_train,
                reward_preds_train,
                _,
                _,
                task_logvars,
            ) = self.evaluate(self.train_tasks[: len(self.eval_tasks)])
            (
                returns_eval,
                success_rate_eval,
                observations_eval,
                rewards_eval,
                reward_preds_eval,
                _,
                _,
                _,
            ) = self.evaluate(self.eval_tasks)
        else:
            (
                returns_eval,
                success_rate_eval,
                _,
                rewards_eval,
                reward_preds_eval,
                _,
                _,
                task_logvars,
            ) = self.evaluate(self.eval_tasks)

        # --- log training  ---
        ## set env steps for tensorboard: z is for lowest order
        logger.record_step(self._n_env_steps_total)
        logger.record_tabular("z/env_steps", self._n_env_steps_total)
        logger.record_tabular("z/time_cost", int(time.time() - self._start_time))
        logger.record_tabular("z/rl_steps", self._n_rl_update_steps_total)
        logger.record_tabular("z/vae_steps", self._n_vae_update_steps_total)
        logger.record_tabular("z/rollouts", self._n_rollouts_total)

        ## gradient norms
        logger.record_tabular(
            "vae_loss/encoder_grad_norm", utl.get_grad_norm(self.vae.encoder)
        )
        if self.vae.reward_decoder is not None:
            logger.record_tabular(
                "vae_loss/reward_decoder_grad_norm",
                utl.get_grad_norm(self.vae.reward_decoder),
            )
        if self.vae.state_decoder is not None:
            logger.record_tabular(
                "vae_loss/state_decoder_grad_norm",
                utl.get_grad_norm(self.vae.state_decoder),
            )

        if len(self.train_tasks) > 0:
            if "plot_behavior" in dir(self.env.unwrapped):  # plot goal-reaching trajs
                for i, task in enumerate(
                    self.train_tasks[: min(5, len(self.eval_tasks))]
                ):
                    self.env.reset(task=task)  # must have task argument
                    logger.add_figure(
                        "trajectory/train_task_{}".format(i),
                        utl_eval.plot_rollouts(observations[i, :], self.env),
                    )

                for i, task in enumerate(
                    self.eval_tasks[: min(5, len(self.eval_tasks))]
                ):
                    self.env.reset(task=task)
                    logger.add_figure(
                        "trajectory/eval_task_{}".format(i),
                        utl_eval.plot_rollouts(observations_eval[i, :], self.env),
                    )

            if "is_goal_state" in dir(
                self.env.unwrapped
            ):  # goal-reaching success rates
                # some metrics
                logger.record_tabular(
                    "metrics/successes_in_buffer",
                    self._successes_in_buffer / self._n_env_steps_total,
                )
                logger.record_tabular(
                    "metrics/success_rate_train", np.mean(success_rate_train)
                )
                logger.record_tabular(
                    "metrics/success_rate_eval", np.mean(success_rate_eval)
                )

            for episode_idx in range(self.max_rollouts_per_task):
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

        # task embedding variance / uncertainty (tasks, T+1, embed_size)
        # multivariate normal entropy ~ log det(Var) = \sum_i log var[i]
        logger.record_tabular(
            "task_uncertainty/ent0_init", task_logvars[:, 0].sum(-1).mean()
        )
        logger.record_tabular(
            "task_uncertainty/ent1_halfway",
            task_logvars[:, int(task_logvars.shape[1] / 2)].sum(-1).mean(),
        )
        logger.record_tabular(
            "task_uncertainty/ent2_final", task_logvars[:, -1].sum(-1).mean()
        )

        logger.dump_tabular()

    def training_mode(self, mode):
        # policy
        self.agent.train(mode)
        # encoder
        self.vae.encoder.train(mode)
        # decoders
        if self.vae.decode_reward:
            self.vae.reward_decoder.train(mode)
        if self.vae.decode_state:
            self.vae.state_decoder.train(mode)

    def _sample_train_task(self):
        if len(self.train_tasks) > 0:
            return self.train_tasks[np.random.randint(len(self.train_tasks))]
        return None

    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False):
        """collect num_rollouts of trajectories in task and save into vae and policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        :return:
        """

        for _ in range(num_rollouts):
            task = self._sample_train_task()  # random sample a training task

            if "task" in inspect.getargspec(self.env.reset)[0]:
                obs = ptu.from_numpy(self.env.reset(task=task))  # reset meta task
            else:
                obs = ptu.from_numpy(self.env.reset())  # reset mdp/pomdp
            obs = obs.reshape(-1, obs.shape[-1])
            done_rollout = False

            # get prior parameters at timestep=0: b_0 = q(m|tau=0)
            _, task_mean, task_logvar, hidden_state = self.reset_encoding()

            # add distribution parameters to observation - policy is conditioned on posterior
            augmented_obs = self.get_augmented_obs(
                obs=obs, task_mu=task_mean, task_std=task_logvar
            )

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor([self.env.action_space.sample()])  # (1, A)
                else:
                    action, _, _, _ = self.agent.act(obs=augmented_obs)  # (1, A)

                # observe reward and next obs (B=1, dim)
                next_obs, reward, done, _ = utl.env_step(
                    self.env, action.squeeze(dim=0)
                )
                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                # belief reward - averaging over multiple latent embeddings - R+(s) = E{m~b}[R(s;m)]
                # this is inaccurate and non-stationary w.r.t. m and decoder params
                if self.belief_reward:
                    belief_reward = self.vae.compute_belief_reward(
                        self.num_belief_samples,
                        task_mean,
                        task_logvar,
                        obs=obs,
                        next_obs=next_obs,
                        actions=action,
                    ).view(-1, 1)

                # update encoding: belief at timestep t
                _, task_mean, task_logvar, hidden_state = self.update_encoding(
                    obs=next_obs,
                    action=action,
                    reward=reward,
                    hidden_state=hidden_state,
                )

                # get augmented next obs
                augmented_next_obs = self.get_augmented_obs(
                    obs=next_obs, task_mu=task_mean, task_std=task_logvar
                )

                # add data to vae buffer - (s, a, r, s', terminal)
                # the only usage of terminal: term must be True iif end for sampling traj
                self.vae_storage.add_sample(
                    observation=ptu.get_numpy(obs.squeeze(dim=0)),
                    action=ptu.get_numpy(action.squeeze(dim=0)),
                    reward=ptu.get_numpy(reward.squeeze(dim=0)),
                    terminal=ptu.get_numpy(done.squeeze(dim=0)),
                    next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                )

                # add data to policy buffer - (s+, a, r, s'+, terminal')
                term = (
                    self.env.unwrapped.is_goal_state()
                    if "is_goal_state" in dir(self.env.unwrapped)
                    else False
                )
                # as long as reaching goal, terminal'=True, train q value with single reward; otherwise False
                self.policy_storage.add_sample(
                    observation=ptu.get_numpy(augmented_obs.squeeze(dim=0)),
                    action=ptu.get_numpy(action.squeeze(dim=0)),
                    reward=ptu.get_numpy(belief_reward.squeeze(dim=0))
                    if self.belief_reward
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

    @torch.no_grad()
    def reset_encoding(self):
        """Basically we don't feed initial obs into encoder...
        set initial hidden state h0 as zeros, and get b0 = fc(h0)
        """
        return self.vae.encoder.prior(batch_size=1)  # all (1, B=1, dim)

    @torch.no_grad()
    def update_encoding(self, obs, action, reward, hidden_state):
        """obs: next_obs s', action: a, reward: r, done: d, hidden_state: h.
        the order: current obs s and hidden state h, we take action a, receive (r,d,s')
        we want to use new transition (a,r,s') and current h to get h', and b'.
        NOTE: compared to BORel and varibad, fix bug to not to reset hidden state if done=True
                  cuz the end belief should still be under current task, and hidden state will be reset in next task
        """
        # all size should be (B, dim)
        return self.vae.encoder(
            actions=action, states=obs, rewards=reward, hidden_state=hidden_state
        )

    def get_augmented_obs(self, obs, task_sample=None, task_mu=None, task_std=None):

        augmented_obs = obs.clone()

        if self.sample_embeddings and (task_sample is not None):  # posterior sampling
            augmented_obs = torch.cat((augmented_obs, task_sample), dim=1)
        elif (task_mu is not None) and (task_std is not None):  # belief state
            task_mu = task_mu.reshape((-1, task_mu.shape[-1]))
            task_std = task_std.reshape((-1, task_std.shape[-1]))
            augmented_obs = torch.cat((augmented_obs, task_mu, task_std), dim=-1)

        return augmented_obs

    def _get_augmented_obs_dim(self):
        dim = self.obs_dim
        if self.sample_embeddings:
            dim += self.vae.task_embedding_size
        else:  # here we assume gaussion, so suff stats are 2x
            dim += 2 * self.vae.task_embedding_size

        return dim

    def sample_rl_batch(self, batch_size):
        # this batch consists of transitions sampled randomly from replay buffer
        batch = self.policy_storage.random_batch(batch_size)
        return ptu.np_to_pytorch_batch(batch)  # all items are (B, dim)

    def sample_vae_batch(self, batch_num_rollouts):
        """sample batch of episodes for vae training"""
        batch = self.vae_storage.random_episodes(batch_num_rollouts, replace=True)
        # sanity check: done[-1] and next_obs[-1,:,-1] are all ones
        return ptu.np_to_pytorch_batch(batch)  # all items are (seq_len, B, dim)

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

        self.training_mode(False)
