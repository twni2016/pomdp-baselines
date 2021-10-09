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
import sunblaze_envs


def train(
    env_id,
    total_episodes,
    policy,
    lr,
    num_processes,
    rew_scale,
    seed,
    nsteps,
    ent_coef,
):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

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
    misc.env_modifiers.make_reward_filtered(env, lambda x: x * rew_scale)

    # Set up policy
    " Note that currently the architecture is fixed for each type of policy regardless of environment "
    if policy == "mlp":
        policy_fn = base.mlp_policy
    elif policy == "lstm":
        # Uses diag. covar. (default)
        policy_fn = base.lstm_policy
    else:
        raise NotImplementedError

    # Run algorithm
    if "Breakout" in env_id or "SpaceInvaders" in env_id:
        raise NotImplementedError
    else:
        a2c_episodes.learn(
            policy=policy_fn,
            env=env,
            nsteps=nsteps,
            # nsteps=5,
            total_episodes=total_episodes,
            max_timesteps=total_episodes * env.venv.envs[0].spec.max_episode_steps,
            lr=lr,
            ent_coef=ent_coef,  # default 0.01 in baselines, 0.0001 in chainer A3C
            lrschedule="linear",
            epsilon=1e-5,
            alpha=0.99,
            gamma=0.99,
            save_interval=100,
        )


def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--env", help="environment ID", default="SunblazeCartPole-v0")
    parser.add_argument("--seed", type=int, help="RNG seed, defaults to random")
    parser.add_argument("--output", type=str)

    # parser.add_argument('--episodes-per-trial', type=int, default=5)
    # parser.add_argument('--trials', type=int, default=10 ** 4)
    # The total number of episodes is now trials*episodes_per_trial
    parser.add_argument("--total-episodes", type=int, default=5e4)

    parser.add_argument(
        "--policy", help="Policy architecture", choices=["mlp", "lstm"], default="mlp"
    )
    parser.add_argument("--processes", default=1, help='int or "max" for all')
    parser.add_argument("--reward-scale", type=float, default=1.0)

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--nsteps", type=int, default=5)
    parser.add_argument("--ent-coef", type=float, default=1e-2)

    args = parser.parse_args()
    # total_episodes = args.trials * args.episodes_per_trial

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
        policy=args.policy,
        lr=args.lr,
        num_processes=ncpu,
        rew_scale=args.reward_scale,
        seed=seed,
        nsteps=args.nsteps,
        ent_coef=args.ent_coef,  # default 0.01 in baselines, 0.0001 in chainer A3C
    )


if __name__ == "__main__":
    main()
