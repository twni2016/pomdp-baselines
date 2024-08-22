# -*- coding: future_fstrings -*-
import sys, os, time

t0 = time.time()
import socket
import numpy as np
import torch
from ruamel.yaml import YAML
from utils import system, logger
import torchkit.pytorch_utils as ptu
import gym

from PPO.a2c_ppo_acktr import algo
from PPO.a2c_ppo_acktr.arguments import get_args
from PPO.a2c_ppo_acktr.envs import make_vec_envs
from PPO.a2c_ppo_acktr.model import Policy
from PPO.a2c_ppo_acktr.storage import RolloutStorage
from PPO.evaluation import evaluate

if __name__ == "__main__":
    args = get_args()

    yaml = YAML()
    v = yaml.load(open(args.config))

    # system: device, threads, seed, pid
    seed = v["seed"]
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)
    system.reproduce(seed)
    pid = str(os.getpid())
    if "SLURM_JOB_ID" in os.environ:
        pid += "_" + str(os.environ["SLURM_JOB_ID"])  # use job id

    # set gpu
    ptu.set_gpu_mode(torch.cuda.is_available() and v["cuda"] >= 0, v["cuda"])

    # logs
    exp_id = "logs/"
    # exp_id = 'debug/'

    env_type = v["env"]["env_type"]
    if len(v["env"]["env_name"].split("-")) == 3:
        # pomdp env: name-{F/P/V}-v0
        env_name, pomdp_type, _ = v["env"]["env_name"].split("-")
        env_name = env_name + "/" + pomdp_type
    else:
        env_name = v["env"]["env_name"]
    exp_id += f"{env_type}/{env_name}/"

    algo_name = v["policy"]["algo_name"]
    assert algo_name in ["a2c", "ppo"] and algo_name == args.algo
    exp_id += f"{algo_name}_gru/"

    os.makedirs(exp_id, exist_ok=True)
    log_folder = os.path.join(exp_id, system.now_str())
    logger_formats = ["stdout", "log", "csv"]
    if v["eval"]["log_tensorboard"]:
        logger_formats.append("tensorboard")
    logger.configure(dir=log_folder, format_strs=logger_formats, precision=4)
    logger.log(f"preload cost {time.time() - t0:.2f}s")

    os.system(f"cp {args.config} {log_folder}/variant_{pid}.yml")
    logger.log(args.config)
    logger.log("pid", pid, socket.gethostname())
    os.makedirs(os.path.join(logger.get_dir(), "save"))

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = ptu.device
    env_name = v["env"]["env_name"]
    num_eval_tasks = v["env"]["num_eval_tasks"]
    if env_type == "rmdp":
        worst_percentile = v["env"]["worst_percentile"]

    envs = make_vec_envs(
        env_type,
        env_name,
        seed,
        args.num_processes,
        args.gamma,
        log_folder,
        device,
        False,
    )

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={"recurrent": args.recurrent_policy},
    )
    actor_critic.to(device)
    logger.log(actor_critic)
    logger.log(envs.observation_space, envs.action_space)

    if algo_name == "a2c":
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm,
        )

    elif algo_name == "ppo":
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
        )

    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
    )
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    start = time.time()
    num_updates = v["train"]["num_iters"]
    # by default, the total env steps per update is 128*16=2048
    logger.log("num_updates", num_updates, args.num_processes)

    for j in range(num_updates):
        t0 = time.time()

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                (
                    value,
                    action,
                    action_log_prob,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                bad_masks,
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value,
            args.use_gae,
            args.gamma,
            args.gae_lambda,
            args.use_proper_time_limits,
        )
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        print("train duration", time.time() - t0)

        if j % v["eval"]["log_interval"] == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            t0 = time.time()

            logger.record_step(total_num_steps)
            logger.record_tabular("z/env_steps", total_num_steps)
            logger.record_tabular("z/time_cost", int(t0 - start))

            eval_episode_rewards, eval_episode_steps = evaluate(
                actor_critic,
                envs.obs_rms,
                env_type,
                env_name,
                seed,
                args.num_processes,
                log_folder,
                device,
                num_eval_tasks,
            )

            if env_type == "pomdp":
                logger.record_tabular(
                    "metrics/total_steps_eval", np.mean(eval_episode_steps)
                )
                logger.record_tabular(
                    "metrics/return_eval_total", np.mean(eval_episode_rewards)
                )
                perf = np.mean(eval_episode_rewards)
            elif env_type == "rmdp":
                cutoff = np.percentile(eval_episode_rewards, 100 * worst_percentile)
                worst_indices = np.where(
                    eval_episode_rewards <= cutoff
                )  # must be "<=" to avoid empty set
                returns_eval_worst, total_steps_eval_worst = (
                    eval_episode_rewards[worst_indices],
                    eval_episode_steps[worst_indices],
                )

                logger.record_tabular(
                    "metrics/return_eval_avg", eval_episode_rewards.mean()
                )
                logger.record_tabular(
                    "metrics/return_eval_worst", returns_eval_worst.mean()
                )
                logger.record_tabular(
                    "metrics/total_steps_eval_avg", eval_episode_steps.mean()
                )
                logger.record_tabular(
                    "metrics/total_steps_eval_worst", total_steps_eval_worst.mean()
                )

            logger.dump_tabular()

            print("eval duration", time.time() - t0)

            # save for every interval-th episode or for the last epoch
            if j % v["eval"]["save_interval"] == 0 and j > 0.75 * num_updates:
                torch.save(
                    [
                        actor_critic.state_dict(),
                        getattr(envs, "obs_rms", None),
                    ],
                    os.path.join(
                        logger.get_dir(), "save", f"agent_{j}_perf{perf:.3f}.pt"
                    ),
                )
