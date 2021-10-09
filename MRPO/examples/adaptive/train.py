import argparse
import os
import random

from baselines import bench, logger
import tensorflow as tf
import gym
from gym.utils.seeding import create_seed
from chainerrl import misc

from . import base
from . import a2c_episodes
from . import ppo2_episodes
import sunblaze_envs


def train(
    env_id,
    trials,
    episodes_per_trial,
    alg,
    policy,
    lr,
    num_processes,
    rew_scale,
    seed,
    nsteps,
    nminibatches,
    akl_coef,
    ent_coef,
):
    from baselines.common import set_global_seeds
    from .vec_env.vec_normalize import VecNormalize
    from .vec_env.dummy_vec_env import DummyVecEnv
    from .vec_env.subproc_vec_env import SubprocVecEnv
    from .bench_monitor import Monitor

    # Set up environment
    ncpu = num_processes
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=ncpu,
        inter_op_parallelism_threads=ncpu,
    )
    tf.Session(config=config).__enter__()

    set_global_seeds(seed)

    if ncpu == 1:

        def make_env():
            env = base.make_env(env_id, outdir=logger.get_dir())
            # Set the env seed
            # (was missing before and is necessary to reproduce runs)
            # Other RNGs handled with 'set_global_seeds'
            env.seed(seed)
            env = Monitor(env, logger.get_dir())
            misc.env_modifiers.make_reward_filtered(env, lambda x: x * rew_scale)
            return env

        env = DummyVecEnv([make_env])
    else:

        def make_env(rank):
            def _thunk():
                env = base.make_env(env_id, process_idx=rank, outdir=logger.get_dir())
                env.seed(seed + rank)
                if logger.get_dir():
                    env = Monitor(
                        env,
                        os.path.join(
                            logger.get_dir(), "train-{}.monitor.json".format(rank)
                        ),
                    )
                misc.env_modifiers.make_reward_filtered(env, lambda x: x * rew_scale)
                return env

            return _thunk

        env = SubprocVecEnv([make_env(i) for i in range(ncpu)])

    env = VecNormalize(env)

    # Set up policy
    " Note that currently the architecture is fixed for each type of policy regardless of environment"
    if policy == "lstm":
        policy_fn = base.lstm_policy
    elif policy == "gru":
        policy_fn = base.gru_policy
    else:
        raise NotImplementedError

    # Run algorithm
    if "Breakout" in env_id or "SpaceInvaders" in env_id:
        raise NotImplementedError
    elif alg == "a2c":
        a2c_episodes.learn(
            policy=policy_fn,
            env=env,
            nsteps=nsteps,
            total_trials=trials,
            episodes_per_trial=episodes_per_trial,
            max_timesteps=trials
            * episodes_per_trial
            * env.venv.envs[0].spec.max_episode_steps,
            lr=lr,
            ent_coef=ent_coef,  # default 0.01 in baselines, 0.0001 in chainer A3C
            lrschedule="linear",
            epsilon=1e-5,
            alpha=0.99,
            gamma=0.99,
            log_interval=1,
            save_interval=100,
        )
    elif alg == "ppo2":
        ppo2_episodes.learn(
            policy=policy_fn,
            env=env,
            # nminibatches needs to be 1 for adaptive, factor it into nsteps
            nsteps=nsteps // nminibatches,
            nminibatches=1,
            # nsteps=512,  # originally
            # nsteps=2048,  # used by ppo2_baselines / original paper
            # nsteps=1024,  # used by ppo_baselines
            # nminibatches=32,  # used by ppo2_baselines
            # nminibatches=64,  # used by ppo_baselines
            total_trials=trials,
            episodes_per_trial=episodes_per_trial,
            akl_coef=akl_coef,
            ent_coef=0.0,
            lr=lr,
            lam=0.97,
            # lam=0.95,
            gamma=0.99,
            noptepochs=10,
            cliprange=0.2,
            log_interval=1,
            save_interval=100,
        )
    else:
        raise NotImplementedError


def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--env", help="environment ID", default="SunblazeCartPole-v0")
    parser.add_argument("--seed", type=int, help="RNG seed, defaults to random")
    parser.add_argument("--output", type=str)

    parser.add_argument("--episodes-per-trial", type=int, default=5)
    parser.add_argument("--trials", type=int, default=1000)

    parser.add_argument(
        "--algorithm",
        help="Training RL algorithm",
        choices=["a2c", "ppo2"],
        default="ppo2",
    )
    parser.add_argument(
        "--policy", help="Policy architecture", choices=["lstm", "gru"], default="lstm"
    )

    # RL algo. yyperparameters
    # default for a2c:
    # parser.add_argument('--lr', type=float, default=7e-4)
    # default for ppo2:
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--nsteps", type=int, default=2048)
    parser.add_argument(
        "--ent-coef", type=float, default=1e-2, help="Only relevant for A2C"
    )
    parser.add_argument(
        "--nminibatches", type=int, default=32, help="Only relevant for PPO2"
    )
    parser.add_argument(
        "--akl-coef", type=float, default=0, help="Only relevant for PPO2"
    )

    parser.add_argument("--processes", default=1, help='int or "max" for all')
    parser.add_argument("--reward-scale", type=float, default=1.0)

    args = parser.parse_args()

    # Configure logger
    if args.output:
        try:
            os.makedirs(args.output)
        except OSError:
            pass

        logger.reset()
        logger.configure(dir=args.output)

    # If seed is unspecified, generate a pseudorandom one
    if not args.seed:
        # "Seed must be between 0 and 2**32 - 1"
        seed = create_seed(args.seed, max_bytes=4)
    else:
        seed = args.seed

    # Log it for reference
    with open(os.path.join(args.output, "seed.txt"), "w") as fout:
        fout.write("%d\n" % seed)

    if args.processes == "max":
        ncpu = multiprocessing.cpu_count()
        # from: https://github.com/openai/baselines/blob/1f8a03f3a62367526f20215188fb5ea4b9ec27e0/baselines/ppo2/run_atari.py#L15
        if sys.platform == "darwin":
            ncpu //= 2
    else:
        try:
            ncpu = int(args.processes)
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid number of processes")

    train(
        args.env,
        trials=args.trials,
        episodes_per_trial=args.episodes_per_trial,
        alg=args.algorithm,
        policy=args.policy,
        lr=args.lr,
        num_processes=ncpu,
        rew_scale=args.reward_scale,
        seed=seed,
        nsteps=args.nsteps,
        nminibatches=args.nminibatches,
        akl_coef=args.akl_coef,
        ent_coef=args.ent_coef,  # default 0.01 in baselines, 0.0001 in chainer A3C
    )


if __name__ == "__main__":
    main()
