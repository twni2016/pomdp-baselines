import argparse
import json
import multiprocessing
import os
import random

from baselines.common import set_global_seeds, tf_util as U
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from chainerrl import misc
import gym
from gym import wrappers
from gym.utils.seeding import create_seed
import numpy as np
import tensorflow as tf

from ..a2c_baselines import base
from ..util import NumpyEncoder


def main():
    """This code allows us to compute a Monte Carlo estimate of the
    reward achieved by a purely random policy. It follows the same
    structure as the usual evaluation code."""

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--env", type=str, default="SunblazeCartPole-v0")
    parser.add_argument("--seed", type=int, help="RNG seed, defaults to random")
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--max-episode-len", type=int, default=10000)
    parser.add_argument("--eval-n-trials", type=int, default=100)
    parser.add_argument("--episodes-per-trial", type=int, default=1)
    parser.add_argument("--eval-n-parallel", type=int, default=1)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    total_episodes = args.eval_n_trials * args.episodes_per_trial

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # If seed is unspecified, generate a pseudorandom one
    if not args.seed:
        # "Seed must be between 0 and 2**32 - 1"
        seed = create_seed(args.seed, max_bytes=4)
    else:
        seed = args.seed

    # Log it for reference
    with open(os.path.join(args.outdir, "seed.txt"), "w") as fout:
        fout.write("%d\n" % seed)

    set_global_seeds(seed)

    output_lock = multiprocessing.Lock()

    def evaluator(process_idx):
        def make_env():
            env = base.make_env(args.env, process_idx)
            env.seed(seed + process_idx)

            if args.record:
                env = gym.wrappers.Monitor(env, args.outdir)
            return env

        env = DummyVecEnv([make_env])

        # if args.record:
        #    env = gym.wrappers.Monitor(env, args.outdir)

        # env = VecNormalize(env)
        obs_space = env.observation_space
        act_space = env.action_space
        if len(act_space.shape) == 0:
            act_shape = (1,)
        else:
            act_shape = (1, act_space.shape[0])

        # Load model
        with U.make_session(num_cpu=1) as sess:
            if "SpaceInvaders" in args.env or "Breakout" in args.env:
                raise NotImplementedError

            # Unwrap DummyVecEnv to access mujoco.py object
            env_base = env.envs[0].unwrapped

            # Record a binary success measure if the env supports it
            if hasattr(env_base, "is_success") and callable(
                getattr(env_base, "is_success")
            ):
                success_support = True
            else:
                print("Warning: env does not support binary success, ignoring.")
                success_support = False

            for _ in range(total_episodes):
                obs = env.reset()
                episode_rew = 0
                success = False
                for _ in range(args.max_episode_len):
                    action = act_space.sample()
                    action = np.reshape(np.asarray(action), act_shape)
                    obs, rew, done, _ = env.step(action)
                    episode_rew += rew
                    if success_support and env_base.is_success():
                        success = True
                    if done:
                        break

                with output_lock:
                    with open(
                        os.path.join(args.outdir, "evaluation.json"), "a"
                    ) as results_file:
                        results_file.write(
                            json.dumps(
                                {
                                    "reward": episode_rew,
                                    "success": success if success_support else "N/A",
                                    "environment": env_base.parameters,
                                    "model": "Random",
                                },
                                cls=NumpyEncoder,
                            )
                        )
                        results_file.write("\n")

    misc.async.run_async(args.eval_n_parallel, evaluator)


if __name__ == "__main__":
    main()
