# -*- coding: future_fstrings -*-
import os, sys
import time

import numpy as np
import torch
import random
import copy

import gym
from .models.policy_rnn import ModelFreeOffPolicy_Separate_RNN as Policy_RNN
from .models.policy_rnn_shared import ModelFreeOffPolicy_Shared_RNN as Policy_Shared_RNN
from .models.policy_mlp import ModelFreeOffPolicy_MLP as Policy_MLP
from buffers.simple_replay_buffer import SimpleReplayBuffer
from buffers.seq_replay_buffer import SeqReplayBuffer

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from utils import evaluation as utl_eval
from utils import logger


class Learner:
    """
    Learner class for SAC/TD3 x MLP/RNN
    usage: a wide range of POMDP environments and corresponding tasks
    """

    def __init__(self, env_args, train_args, eval_args, policy_args, seed, **kwargs):
        self.seed = seed

        self.init_env(**env_args)

        self.init_policy(**policy_args)

        self.init_train(**train_args)

        self.init_eval(**eval_args)

    def init_env(
        self,
        env_type,
        env_name,
        max_rollouts_per_task=None,
        num_tasks=None,
        num_train_tasks=None,
        num_eval_tasks=None,
        report_eval_tasks=None,
        eval_envs=None,
        worst_percentile=None,
        **kwargs
    ):

        # initialize environment
        assert env_type in ["meta", "pomdp", "rmdp", "metaworld", "generalize"]
        self.env_type = env_type

        if self.env_type == "meta":  # meta tasks: using varibad wrapper
            from envs.meta.make_env import make_env

            self.train_env = make_env(
                env_name,
                max_rollouts_per_task,
                seed=self.seed,
                n_tasks=num_tasks,
                **kwargs
            )
            self.eval_env = copy.deepcopy(self.train_env)
            # unwrapped env to get some info about the environment
            unwrapped_env = self.train_env.unwrapped

            # split to train/eval tasks
            assert num_train_tasks >= num_eval_tasks > 0
            shuffled_tasks = np.random.permutation(unwrapped_env.get_all_task_idx())
            self.train_tasks = shuffled_tasks[:num_train_tasks]
            self.eval_tasks = shuffled_tasks[-num_eval_tasks:]

            # calculate what the maximum length of the trajectories is
            self.max_rollouts_per_task = max_rollouts_per_task
            self.max_trajectory_len = self.train_env.horizon_bamdp  # H^+ = k * H

        elif self.env_type == "pomdp":  # pomdp/mdp task, using pomdp wrapper
            import envs.pomdp

            assert num_eval_tasks > 0
            self.train_env = gym.make(env_name)
            self.eval_env = copy.deepcopy(self.train_env)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        elif self.env_type == "rmdp":  # robust mdp task, using robust mdp wrapper
            sys.path.append("envs/rl-generalization")
            import sunblaze_envs

            assert (
                num_eval_tasks > 0 and worst_percentile > 0.0 and worst_percentile < 1.0
            )
            self.train_env = sunblaze_envs.make(env_name)
            assert np.all(self.train_env.action_space.low == -1)
            assert np.all(self.train_env.action_space.high == 1)
            self.eval_env = copy.deepcopy(self.train_env)  # same as train_env
            self.worst_percentile = worst_percentile

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        elif self.env_type == "metaworld":
            # Now we only support MetaWorld ML10 and ML45 benchmarks
            assert env_name in ["ML10", "ML45"]
            assert num_train_tasks in [10, 45]
            assert num_eval_tasks == 5
            import metaworld

            if env_name == "ML10":
                self.benchmark = metaworld.ML10(seed=self.seed)
            else:
                self.benchmark = metaworld.ML45(seed=self.seed)
            for name, env_cls in self.benchmark.train_classes.items():
                self.train_env = env_cls()
                # NOTE: we temporarily assign self.train_env for common interface but won't use it later
                break
            # import ipdb; ipdb.set_trace()
            self.num_train_tasks = num_train_tasks
            self.num_eval_tasks = num_eval_tasks
            self.max_rollouts_per_task = 1
            self.max_trajectory_len = (
                self.train_env.max_path_length
            )  # constant for metaworld
            assert self.max_trajectory_len == 500

        elif self.env_type == "generalize":
            sys.path.append("envs/rl-generalization")
            import sunblaze_envs

            self.train_env = sunblaze_envs.make(env_name)
            assert np.all(self.train_env.action_space.low == -1)
            assert np.all(self.train_env.action_space.high == 1)

            def check_env_class(env_name):
                if "Normal" in env_name:
                    return "R"
                if "Extreme" in env_name:
                    return "E"
                return "D"

            self.train_env_name = check_env_class(env_name)

            self.eval_envs = {
                sunblaze_envs.make(env_name): (check_env_class(env_name), num_eval_task)
                for env_name, num_eval_task in eval_envs.items()
            }  # several types of evaluation envs
            logger.log(self.train_env_name, self.train_env)
            logger.log(self.eval_envs)

            self.train_tasks = []
            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        else:
            raise ValueError

        # get action / observation dimensions
        assert self.train_env.action_space.__class__.__name__ == "Box"
        self.act_dim = self.train_env.action_space.shape[0]
        self.obs_dim = self.train_env.observation_space.shape[0]  # include 1-dim done
        logger.log("obs_dim", self.obs_dim, "act_dim", self.act_dim)

    def init_policy(self, arch, separate: bool = True, **kwargs):
        # initialize policy
        assert arch in ["mlp", "lstm", "gru"]
        self.policy_arch = arch
        if arch == "mlp":
            agent_class = Policy_MLP
        elif separate == True:
            agent_class = Policy_RNN
        else:
            agent_class = Policy_Shared_RNN
            logger.log("WARNING: YOU ARE USING SHARED ACTOR-CRITIC ARCH !!!!!!!")
        self.agent = agent_class(
            encoder=arch,  # redundant for Policy_MLP
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            **kwargs
        ).to(ptu.device)
        logger.log(self.agent)

    def init_train(
        self,
        buffer_size,
        batch_size,
        num_iters,
        num_init_rollouts_pool,
        num_rollouts_per_iter,
        num_updates_per_iter=None,
        sampled_seq_len=None,
        sample_weight_baseline=None,
        **kwargs
    ):

        if num_updates_per_iter is None:
            num_updates_per_iter = 1.0
        assert isinstance(num_updates_per_iter, int) or isinstance(
            num_updates_per_iter, float
        )
        # if int, it means absolute value; if float, it means the multiplier of collected env steps
        self.num_updates_per_iter = num_updates_per_iter

        if self.policy_arch == "mlp":
            self.policy_storage = SimpleReplayBuffer(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim,
                max_trajectory_len=self.max_trajectory_len,
                add_timeout=False,  # no timeout storage
            )

        else:  # rnn
            if sampled_seq_len == -1:
                sampled_seq_len = self.max_trajectory_len

            self.policy_storage = SeqReplayBuffer(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim,
                sampled_seq_len=sampled_seq_len,
                sample_weight_baseline=sample_weight_baseline,
            )

        self.batch_size = batch_size
        self.num_iters = num_iters
        self.num_init_rollouts_pool = num_init_rollouts_pool
        self.num_rollouts_per_iter = num_rollouts_per_iter
        if self.env_type == "metaworld":
            assert (
                num_init_rollouts_pool == num_rollouts_per_iter == self.num_train_tasks
            )

        total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
        self.n_env_steps_total = self.max_trajectory_len * total_rollouts
        logger.log(
            "*** total rollouts",
            total_rollouts,
            "total env steps",
            self.n_env_steps_total,
        )

    def init_eval(
        self,
        log_interval,
        save_interval,
        log_tensorboard,
        eval_stochastic=False,
        num_episodes_per_task=1,
        **kwargs
    ):

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.log_tensorboard = log_tensorboard
        self.eval_stochastic = eval_stochastic
        self.eval_num_episodes_per_task = num_episodes_per_task

    def train(self):
        """
        training loop
        NOTE: the main difference from BORel to varibad is changing the alternation
        of rollout collection and model updates: in varibad, one step collection is
        followed by several model updates; while in borel, we collect several **entire**
        trajectories from random sampled tasks for each model updates.
        """

        self._start_training()

        if self.num_init_rollouts_pool > 0:
            logger.log("Collecting initial pool of data..")
            while (
                self._n_env_steps_total
                < self.num_init_rollouts_pool * self.max_trajectory_len
            ):
                self.collect_rollouts(  # to make sure metaworld we sample a suite
                    num_rollouts=self.num_init_rollouts_pool
                    if self.env_type == "metaworld"
                    else 1,
                    random_actions=True,
                )
            logger.log(
                "Done! env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            if isinstance(self.num_updates_per_iter, float):
                # update: pomdp task updates more for the first iter_
                train_stats = self.update(
                    int(self._n_env_steps_total * self.num_updates_per_iter)
                )
                self.log_train_stats(train_stats)

        last_eval_num_iters = 0
        while self._n_env_steps_total < self.n_env_steps_total:
            # collect data from num_rollouts_per_iter train tasks:
            env_steps = self.collect_rollouts(num_rollouts=self.num_rollouts_per_iter)
            logger.log("env steps", self._n_env_steps_total)

            train_stats = self.update(
                self.num_updates_per_iter
                if isinstance(self.num_updates_per_iter, int)
                else int(self.num_updates_per_iter * env_steps)
            )
            self.log_train_stats(train_stats)

            # evaluate and log
            current_num_iters = self._n_env_steps_total // (
                self.num_rollouts_per_iter * self.max_trajectory_len
            )
            if (
                current_num_iters != last_eval_num_iters
                and current_num_iters % self.log_interval == 0
            ):
                last_eval_num_iters = current_num_iters
                perf = self.log()
                if (
                    self._n_env_steps_total > 0.75 * self.n_env_steps_total
                    and current_num_iters % self.save_interval == 0
                ):
                    # save models in later training stage
                    self.save_model(current_num_iters, perf)
        self.save_model(current_num_iters, perf)

    def update(self, num_updates):
        rl_losses_agg = {}
        # print(num_updates)
        for update in range(num_updates):
            # sample random RL batch: in transitions
            batch = self.sample_rl_batch(self.batch_size)

            # RL update
            # t0 = time.time()
            rl_losses = self.agent.update(batch)
            # print("train", time.time() - t0)

            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += num_updates

        return rl_losses_agg

    @torch.no_grad()
    def evaluate(self, tasks, deterministic=True):
        if self.env_type == "metaworld":
            assert tasks in ["train", "test"]
            if tasks == "train":
                env_classes = self.benchmark.train_classes
                env_tasks = self.benchmark.train_tasks
            else:
                env_classes = self.benchmark.test_classes
                env_tasks = self.benchmark.test_tasks
            envs = []
            for name, env_cls in env_classes.items():
                env = env_cls()
                # random sample **one** task for this env
                task = random.choice(
                    [task for task in env_tasks if task.env_name == name]
                )
                env.set_task(task)  # must set here
                envs.append(env)
            tasks = envs  # HACK: bad coding practice

        num_episodes = self.max_rollouts_per_task  # k
        # max_trajectory_len = k*H
        returns_per_episode = np.zeros((len(tasks), num_episodes))
        success_rate = np.zeros(len(tasks))
        total_steps = np.zeros(len(tasks))

        if self.env_type == "meta":
            num_steps_per_episode = self.eval_env.unwrapped._max_episode_steps  # H
            obs_size = self.eval_env.unwrapped.observation_space.shape[
                0
            ]  # original size
            observations = np.zeros((len(tasks), self.max_trajectory_len + 1, obs_size))
        elif self.env_type == "metaworld":
            num_steps_per_episode = self.max_trajectory_len
            observations = None
        else:  # pomdp, rmdp, generalize
            num_steps_per_episode = self.eval_env._max_episode_steps
            observations = None

        for task_idx, task in enumerate(tasks):
            step = 0
            if self.env_type == "metaworld":
                self.eval_env = task
                print(self.eval_env)

            if self.env_type == "meta":
                obs = ptu.from_numpy(self.eval_env.reset(task=task))  # reset task
                observations[task_idx, step, :] = ptu.get_numpy(obs[:obs_size])
            else:
                obs = ptu.from_numpy(self.eval_env.reset())  # reset

            obs = obs.reshape(1, obs.shape[-1])

            if self.policy_arch in ["lstm", "gru"]:
                action, reward, internal_state = self.agent.get_initial_info()

            for episode_idx in range(num_episodes):
                running_reward = 0.0
                for _ in range(num_steps_per_episode):
                    if self.policy_arch == "mlp":
                        action, _, _, _ = self.agent.act(
                            obs, deterministic=deterministic
                        )
                    else:
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=obs,
                            deterministic=deterministic,
                        )

                    # observe reward and next obs
                    next_obs, reward, done, info = utl.env_step(
                        self.eval_env, action.squeeze(dim=0)
                    )
                    running_reward += reward.item()
                    step += 1
                    done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                    if self.env_type == "meta":
                        observations[task_idx, step, :] = ptu.get_numpy(
                            next_obs[0, :obs_size]
                        )

                    if (
                        self.env_type == "meta"
                        and "is_goal_state" in dir(self.eval_env.unwrapped)
                        and self.eval_env.unwrapped.is_goal_state()
                    ):
                        success_rate[task_idx] = 1.0  # ever once reach
                    if self.env_type == "metaworld" and info["success"] == True:
                        success_rate[task_idx] = 1.0  # ever once reach
                    if (
                        self.env_type == "generalize"
                        and self.eval_env.unwrapped.is_success()
                    ):
                        success_rate[task_idx] = 1.0  # ever once reach

                    if done_rollout and self.env_type in [
                        "rmdp",
                        "pomdp",
                        "generalize",
                    ]:
                        # NOTE: rmdp, pomdp (single episode): early stop (fail), then break;
                        # meta: fixed horizon, ignore the done
                        break

                    # set: obs <- next_obs
                    obs = next_obs.clone()

                returns_per_episode[task_idx, episode_idx] = running_reward
            total_steps[task_idx] = step
        return returns_per_episode, success_rate, observations, total_steps

    def log_train_stats(self, train_stats):
        logger.record_step(self._n_env_steps_total)
        ## log losses
        for k, v in train_stats.items():
            logger.record_tabular("rl_loss/" + k, v)
        ## gradient norms
        if self.policy_arch in ["lstm", "gru"]:
            results = self.agent.report_grad_norm()
            for k, v in results.items():
                logger.record_tabular("rl_loss/" + k, v)
        logger.dump_tabular()

    def log(self):
        # --- log training  ---
        ## set env steps for tensorboard: z is for lowest order
        logger.record_step(self._n_env_steps_total)
        logger.record_tabular("z/env_steps", self._n_env_steps_total)
        logger.record_tabular("z/rollouts", self._n_rollouts_total)
        logger.record_tabular("z/rl_steps", self._n_rl_update_steps_total)

        # --- evaluation ----
        if self.env_type == "meta":
            (
                returns_train,
                success_rate_train,
                observations,
                total_steps_train,
            ) = self.evaluate(self.train_tasks[: len(self.eval_tasks)])
            (
                returns_eval,
                success_rate_eval,
                observations_eval,
                total_steps_eval,
            ) = self.evaluate(self.eval_tasks)
            if self.eval_stochastic:
                (
                    returns_eval_sto,
                    success_rate_eval_sto,
                    observations_eval_sto,
                    total_steps_eval_sto,
                ) = self.evaluate(self.eval_tasks, deterministic=False)

            if "plot_behavior" in dir(
                self.eval_env.unwrapped
            ):  # plot goal-reaching trajs
                for i, task in enumerate(
                    self.train_tasks[: min(5, len(self.eval_tasks))]
                ):
                    self.eval_env.reset(task=task)  # must have task argument
                    logger.add_figure(
                        "trajectory/train_task_{}".format(i),
                        utl_eval.plot_rollouts(observations[i, :], self.eval_env),
                    )

                for i, task in enumerate(
                    self.eval_tasks[: min(5, len(self.eval_tasks))]
                ):
                    self.eval_env.reset(task=task)
                    logger.add_figure(
                        "trajectory/eval_task_{}".format(i),
                        utl_eval.plot_rollouts(observations_eval[i, :], self.eval_env),
                    )
                    if self.eval_stochastic:
                        logger.add_figure(
                            "trajectory/eval_task_{}_sto".format(i),
                            utl_eval.plot_rollouts(
                                observations_eval_sto[i, :], self.eval_env
                            ),
                        )

            if "is_goal_state" in dir(
                self.eval_env.unwrapped
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
                if self.eval_stochastic:
                    logger.record_tabular(
                        "metrics/success_rate_eval_sto", np.mean(success_rate_eval_sto)
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
                if self.eval_stochastic:
                    logger.record_tabular(
                        "metrics/return_eval_episode_{}_sto".format(episode_idx + 1),
                        np.mean(returns_eval_sto[:, episode_idx]),
                    )

            logger.record_tabular(
                "metrics/total_steps_train", np.mean(total_steps_train)
            )
            logger.record_tabular(
                "metrics/return_train_total", np.mean(np.sum(returns_train, axis=-1))
            )
            logger.record_tabular("metrics/total_steps_eval", np.mean(total_steps_eval))
            logger.record_tabular(
                "metrics/return_eval_total", np.mean(np.sum(returns_eval, axis=-1))
            )
            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/total_steps_eval_sto", np.mean(total_steps_eval_sto)
                )
                logger.record_tabular(
                    "metrics/return_eval_total_sto",
                    np.mean(np.sum(returns_eval_sto, axis=-1)),
                )

        elif self.env_type == "metaworld":
            returns_train, success_rate_train = [], []
            returns_eval, success_rate_eval = [], []
            for _ in range(self.eval_num_episodes_per_task):
                return_train, success_train, _, _ = self.evaluate("train")
                return_eval, success_eval, _, _ = self.evaluate("test")
                returns_train.append(return_train)
                success_rate_train.append(success_train)
                returns_eval.append(return_eval)
                success_rate_eval.append(success_eval)
            returns_train = np.stack(returns_train).mean(0).squeeze(-1)  # (Envs)
            success_rate_train = np.stack(success_rate_train).mean(0)  # (Envs)
            returns_eval = np.stack(returns_eval).mean(0).squeeze(-1)
            success_rate_eval = np.stack(success_rate_eval).mean(0)

            logger.record_tabular("metrics/return_train_avg", np.mean(returns_train))
            logger.record_tabular("metrics/succ_train_avg", np.mean(success_rate_train))
            for name, return_train, succ_train in zip(
                self.benchmark.train_classes.keys(), returns_train, success_rate_train
            ):  # order-preserve mapping
                logger.record_tabular(f"metrics/return_train_{name}", return_train)
                logger.record_tabular(f"metrics/succ_train_{name}", succ_train)

            logger.record_tabular("metrics/return_eval_avg", np.mean(returns_eval))
            logger.record_tabular("metrics/succ_eval_avg", np.mean(success_rate_eval))
            for name, return_eval, succ_eval in zip(
                self.benchmark.test_classes.keys(), returns_eval, success_rate_eval
            ):  # order-preserve mapping
                logger.record_tabular(f"metrics/return_eval_{name}", return_eval)
                logger.record_tabular(f"metrics/succ_eval_{name}", succ_eval)

        elif self.env_type == "generalize":
            returns_eval, success_rate_eval, total_steps_eval = {}, {}, {}
            for env, (env_name, eval_num_episodes_per_task) in self.eval_envs.items():
                self.eval_env = env  # assign eval_env, not train_env
                for suffix, deterministic in zip(["", "_sto"], [True, False]):
                    if deterministic == False and self.eval_stochastic == False:
                        continue
                    return_eval, success_eval, _, total_step_eval = self.evaluate(
                        eval_num_episodes_per_task * [None],
                        deterministic=deterministic,
                    )
                    returns_eval[
                        self.train_env_name + env_name + suffix
                    ] = return_eval.squeeze(-1)
                    success_rate_eval[
                        self.train_env_name + env_name + suffix
                    ] = success_eval
                    total_steps_eval[
                        self.train_env_name + env_name + suffix
                    ] = total_step_eval

            for k, v in returns_eval.items():
                logger.record_tabular(f"metrics/return_eval_{k}", np.mean(v))
            for k, v in success_rate_eval.items():
                logger.record_tabular(f"metrics/succ_eval_{k}", np.mean(v))
            for k, v in total_steps_eval.items():
                logger.record_tabular(f"metrics/total_steps_eval_{k}", np.mean(v))

        elif self.env_type == "rmdp":
            returns_eval, _, _, total_steps_eval = self.evaluate(self.eval_tasks)
            returns_eval = returns_eval.squeeze(-1)
            # np.quantile is introduced in np v1.15, so we have to use np.percentile
            cutoff = np.percentile(returns_eval, 100 * self.worst_percentile)
            worst_indices = np.where(
                returns_eval <= cutoff
            )  # must be "<=" to avoid empty set
            returns_eval_worst, total_steps_eval_worst = (
                returns_eval[worst_indices],
                total_steps_eval[worst_indices],
            )

            logger.record_tabular("metrics/return_eval_avg", returns_eval.mean())
            logger.record_tabular(
                "metrics/return_eval_worst", returns_eval_worst.mean()
            )
            logger.record_tabular(
                "metrics/total_steps_eval_avg", total_steps_eval.mean()
            )
            logger.record_tabular(
                "metrics/total_steps_eval_worst", total_steps_eval_worst.mean()
            )

        elif self.env_type == "pomdp":
            returns_eval, success_rate_eval, _, total_steps_eval = self.evaluate(
                self.eval_tasks
            )
            if self.eval_stochastic:
                (
                    returns_eval_sto,
                    success_rate_eval_sto,
                    _,
                    total_steps_eval_sto,
                ) = self.evaluate(self.eval_tasks, deterministic=False)

            logger.record_tabular("metrics/total_steps_eval", np.mean(total_steps_eval))
            logger.record_tabular(
                "metrics/return_eval_total", np.mean(np.sum(returns_eval, axis=-1))
            )
            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/total_steps_eval_sto", np.mean(total_steps_eval_sto)
                )
                logger.record_tabular(
                    "metrics/return_eval_total_sto",
                    np.mean(np.sum(returns_eval_sto, axis=-1)),
                )

        else:
            raise ValueError

        logger.record_tabular("z/time_cost", int(time.time() - self._start_time))
        logger.record_tabular(
            "z/fps",
            (self._n_env_steps_total - self._n_env_steps_total_last)
            / (time.time() - self._start_time_last),
        )
        self._n_env_steps_total_last = self._n_env_steps_total
        self._start_time_last = time.time()

        logger.dump_tabular()

        # import ipdb; ipdb.set_trace()
        if self.env_type == "metaworld":
            return np.mean(success_rate_eval)  # succ is more important
        elif self.env_type == "generalize":
            return sum([v.mean() for v in success_rate_eval.values()]) / len(
                success_rate_eval
            )
        else:
            return np.mean(np.sum(returns_eval, axis=-1))

    def save_model(self, iter, perf):
        save_path = os.path.join(
            logger.get_dir(), "save", f"agent_{iter}_perf{perf:.3f}.pt"
        )
        torch.save(self.agent.state_dict(), save_path)

    def _sample_train_task(self):
        if len(self.train_tasks) > 0:
            return self.train_tasks[np.random.randint(len(self.train_tasks))]
        return None

    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """
        if self.env_type == "metaworld":
            assert num_rollouts == len(self.benchmark.train_classes)
            training_envs = []
            for name, env_cls in self.benchmark.train_classes.items():
                env = env_cls()
                # random sample a task for this training env
                task = random.choice(
                    [
                        task
                        for task in self.benchmark.train_tasks
                        if task.env_name == name
                    ]
                )
                env.set_task(task)  # must set here
                training_envs.append(env)

        before_env_steps = self._n_env_steps_total
        for idx in range(num_rollouts):
            steps = 0

            if self.env_type == "metaworld":
                self.train_env = training_envs[idx]
                print(self.train_env)
            else:
                task = self._sample_train_task()  # random sample a training task
            if self.env_type == "meta":
                obs = ptu.from_numpy(self.train_env.reset(task=task))  # reset task
            else:  # pomdp, rmdp, generalize, metaworld
                obs = ptu.from_numpy(self.train_env.reset())  # reset

            obs = obs.reshape(1, obs.shape[-1])
            done_rollout = False

            # get hidden state at timestep=0, None for mlp
            if self.policy_arch in ["lstm", "gru"]:
                action, reward, internal_state = self.agent.get_initial_info()
                # temporary storage
                obs_list, act_list, rew_list, next_obs_list, term_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor(
                        [self.train_env.action_space.sample()]
                    )  # (1, A)
                else:  # policy takes hidden state as input for rnn, while takes obs for mlp
                    if self.policy_arch == "mlp":
                        action, _, _, _ = self.agent.act(obs, deterministic=False)
                    else:
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=obs,
                            deterministic=False,
                        )

                # observe reward and next obs (B=1, dim)
                next_obs, reward, done, info = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )
                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                # update statistics
                steps += 1

                # add data to policy buffer - (s+, a, r, s'+, term')
                if self.env_type == "meta" and "is_goal_state" in dir(
                    self.train_env.unwrapped
                ):
                    # NOTE: following varibad practice: for meta env, even if reaching the goal (term=True),
                    # the episode still continues.
                    term = self.train_env.unwrapped.is_goal_state()
                    self._successes_in_buffer += int(term)
                elif self.env_type == "metaworld":
                    term = False  # generalize tasks done = False always
                    # self._successes_in_buffer += int(info['success'])
                else:
                    # early stopping env: such as rmdp, pomdp, generalize tasks. term ignores timeout
                    term = (
                        False
                        if "TimeLimit.truncated" in info
                        or steps >= self.max_trajectory_len
                        else done_rollout
                    )

                if self.policy_arch == "mlp":
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs.squeeze(dim=0)),
                        action=ptu.get_numpy(action.squeeze(dim=0)),
                        reward=ptu.get_numpy(reward.squeeze(dim=0)),
                        terminal=np.array([term], dtype=float),
                        next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                    )
                else:  # append tensors to temporary storage
                    obs_list.append(obs)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)

                # set: obs <- next_obs
                obs = next_obs.clone()
                if (
                    self.env_type == "metaworld"
                    and steps >= self.train_env.max_path_length
                ):
                    break  # has to manually break

            if self.policy_arch in ["lstm", "gru"]:  # add collected sequence to buffer
                self.policy_storage.add_episode(
                    observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                    actions=ptu.get_numpy(torch.cat(act_list, dim=0)),  # (L, dim)
                    rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                    terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                    next_observations=ptu.get_numpy(
                        torch.cat(next_obs_list, dim=0)
                    ),  # (L, dim)
                )

            print(steps, "term", term)
            self._n_env_steps_total += steps
            self._n_rollouts_total += 1
        return self._n_env_steps_total - before_env_steps

    def sample_rl_batch(self, batch_size):
        """sample batch of episodes for vae training"""
        if self.policy_arch == "mlp":
            batch = self.policy_storage.random_batch(batch_size)
        else:  # rnn: # all items are (sampled_seq_len, B, dim)
            batch = self.policy_storage.random_episodes(batch_size)
        # import ipdb; ipdb.set_trace()
        return ptu.np_to_pytorch_batch(batch)

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_env_steps_total_last = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._successes_in_buffer = 0

        self._start_time = time.time()
        self._start_time_last = time.time()
