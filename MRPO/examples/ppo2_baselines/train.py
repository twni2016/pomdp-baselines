import argparse
import os
import random

from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
from gym.utils.seeding import create_seed

from . import base
from . import ppo2_episodes

import sunblaze_envs


def train(
    env_id,
    total_episodes,
    seed,
    ncpu,
    policy,
    lr,
    nsteps,
    nminibatches,
):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize

    # from baselines.ppo2 import ppo2
    # from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

    # Set up environment
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=ncpu,
        inter_op_parallelism_threads=ncpu,
    )
    tf.Session(config=config).__enter__()

    if ncpu == 1:
        # from: https://github.com/openai/baselines/blob/1f8a03f3a62367526f20215188fb5ea4b9ec27e0/baselines/ppo2/run_mujoco.py#L14
        def make_env():
            env = base.make_env(env_id, outdir=logger.get_dir())
            # Set the env seed
            # (was missing before and is necessary to reproduce runs)
            # Other RNGs handled with 'set_global_seeds'
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir())
            return env

        env = DummyVecEnv([make_env])
    else:

        def make_env(rank):
            def _thunk():
                env = base.make_env(env_id, process_idx=rank, outdir=logger.get_dir())
                env.seed(seed + rank)

                if logger.get_dir():
                    env = bench.Monitor(
                        env,
                        os.path.join(
                            logger.get_dir(), "train-{}.monitor.json".format(rank)
                        ),
                    )

                return env

            return _thunk

        env = SubprocVecEnv([make_env(i) for i in range(ncpu)])
    env = VecNormalize(env)

    """
    # Set inside set_global_seeds:
    import numpy as np; import random
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    # Other possible RNGs (don't seem to matter):
    gym.utils.seeding.np_random(seed)
    from gym import spaces
    spaces.prng.seed(seed)
    """

    set_global_seeds(seed)

    # Set up policy
    " Note that currently the architecture is fixed for each type of policy regardless of environment "
    if policy == "mlp":
        policy_fn = base.mlp_policy
    elif policy == "lstm":
        # Uses diag. covar. (default)
        policy_fn = base.lstm_policy
    else:
        raise NotImplementedError

    if "Breakout" in env_id or "SpaceInvaders" in env_id:
        raise NotImplementedError
    else:
        # from https://github.com/openai/baselines/blob/master/baselines/ppo2/run_mujoco.py
        ppo2_episodes.learn(
            # functions the same as old pposgd checkpointer
            save_interval=10,
            # PPO2 mujoco defaults
            policy=policy_fn,
            env=env,
            nsteps=nsteps,
            nminibatches=nminibatches,
            # nsteps=2048,  # used by ppo2_baselines / original paper
            # nsteps=1024,  # used by ppo_baselines
            # nminibatches=32,  # used by ppo2_baselines
            # nminibatches=64,  # used by ppo_baselines
            lam=0.95,
            gamma=0.99,
            noptepochs=10,
            log_interval=1,
            ent_coef=0.0,
            lr=lr,  # 3e-4 is default in ppo and ppo2
            cliprange=0.2,
            total_episodes=total_episodes,
        )
    # closed inside ppo2.learn
    # env.close()


def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--env", help="environment ID", default="SunblazeBreakout-v0")
    parser.add_argument("--seed", type=int, help="RNG seed, defaults to random")
    parser.add_argument("--output", type=str)
    parser.add_argument("--processes", default=1, help='int or "max" for all')
    # parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument("--total-episodes", type=int, default=int(5e4))
    parser.add_argument(
        "--policy", help="Policy architecture", choices=["mlp", "lstm"], default="mlp"
    )

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--nsteps", type=int, default=2048)
    parser.add_argument("--nminibatches", type=int, default=32)

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
        total_episodes=args.total_episodes,
        seed=seed,
        ncpu=ncpu,
        policy=args.policy,
        lr=args.lr,
        nsteps=args.nsteps,
        nminibatches=args.nminibatches,
    )


if __name__ == "__main__":
    main()
