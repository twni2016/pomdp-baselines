import argparse
import json
import multiprocessing
import os
import pickle
import random
import time

from baselines.common import set_global_seeds, tf_util as U
from chainerrl import misc
import gym
from gym.utils.seeding import create_seed

# from gym import wrappers
import numpy as np
import tensorflow as tf

from . import base
from .monitor import AdaptiveVideoMonitor as VideoMonitor
from .a2c_episodes import Model as a2cModel
from .ppo2_episodes import Model as ppoModel
from .vec_env.dummy_vec_env import DummyVecEnv
from .vec_env.vec_normalize import VecNormalize
from ..util import NumpyEncoder


def main():
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('load', type=str)
    parser.add_argument("--normalize", type=str)
    parser.add_argument("--env", type=str, default="SunblazeCartPole-v0")
    parser.add_argument("--seed", type=int, help="RNG seed, defaults to random")
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--max-episode-len", type=int, default=10000)
    parser.add_argument("--eval-n-trials", type=int, default=100)
    parser.add_argument("--episodes-per-trial", type=int, default=5)
    parser.add_argument("--eval-n-parallel", type=int, default=1)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("load", type=str, nargs="*")
    args = parser.parse_args()

    # Fixes problem of eval script being run with ".../checkpoints/*"
    if len(args.load) > 1:
        import natsort

        print("Detected multiple model file args, sorting and choosing last..")
        # Fixes issue of 'normalize' file inside checkpoint folder
        args.load = [f for f in args.load if "normalize" not in f]
        args.load = natsort.natsorted(args.load, reverse=True)[0]
        print("Using {}".format(args.load))
    else:
        args.load = args.load[0]

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
                env = VideoMonitor(env, args.outdir, video_callable=lambda _: True)
            return env

        env = DummyVecEnv([make_env])
        obs_space = env.observation_space
        act_space = env.action_space
        if len(act_space.shape) == 0:
            discrete = True
        else:
            discrete = False

        # TODO(cpacker): this should really be in the top-level dir
        norm_path = (
            args.normalize
            if args.normalize
            else os.path.join(os.path.dirname(args.load), "normalize")
        )
        with open(norm_path, "rb") as f:
            obs_norms = pickle.load(f)
        clipob = obs_norms["clipob"]
        mean = obs_norms["mean"]
        var = obs_norms["var"]

        # Load model
        with U.make_session(num_cpu=1) as sess:
            if "SpaceInvaders" in args.env or "Breakout" in args.env:
                raise NotImplementedError
            else:
                # '.../checkpoint/XXXX' -> '.../make_model.pkl'
                pkl_path = os.path.join(
                    os.path.dirname(os.path.dirname(args.load)), "make_model.pkl"
                )
                # from: https://github.com/openai/baselines/issues/115
                print("[pidx %d] Constructing model from %s" % (process_idx, pkl_path))
                with open(pkl_path, "rb") as fh:
                    import cloudpickle

                    make_model = cloudpickle.load(fh)
                model = make_model()
                print(
                    "[pidx %d] Loading saved model from %s" % (process_idx, args.load)
                )
                model.load(args.load)

            # Unwrap DummyVecEnv to access mujoco.py object
            env_base = env.envs[0].unwrapped

            # Record a binary success measure if the env supports it
            if hasattr(env_base, "is_success") and callable(
                getattr(env_base, "is_success")
            ):
                success_support = True
            else:
                print(
                    "[pidx %d] Warning: env does not support binary success, ignoring."
                    % process_idx
                )

            start = time.time()
            for t in range(args.eval_n_trials):
                progress_pct = 10
                if t > 0 and (
                    (args.eval_n_trials < progress_pct)
                    or (t % (args.eval_n_trials // 10) == 0)
                ):
                    # Indicate progress every 10%
                    elapsed = time.time() - start
                    hours, rem = divmod(elapsed, 3600)
                    minutes, seconds = divmod(rem, 60)
                    print(
                        "[pidx %d] Trial %d/%d, elapsed: %d:%d:%d"
                        % (process_idx, t, args.eval_n_trials, hours, minutes, seconds)
                    )

                obs = env.reset([True])
                state = model.initial_state
                if discrete:
                    action = -1
                    shape = (1,)
                else:
                    shape = (act_space.shape[0],)
                    action = np.zeros(shape, dtype=np.float32)

                rew = 0.0
                done = False
                mask = np.asarray([False])
                success = False
                # Reward for the specific episode in the trial
                all_episodes_rew = np.zeros(args.episodes_per_trial)

                for i in range(args.episodes_per_trial):
                    for _ in range(args.max_episode_len):
                        obs = np.clip((obs - mean) / np.sqrt(var), -clipob, clipob)
                        action = np.reshape(np.asarray([action]), shape)
                        action, value, state, _ = model.step(
                            obs,
                            state,
                            action,
                            np.reshape(np.asarray([rew]), (1,)),
                            np.reshape(np.asarray([done]), (1,)),
                            mask,
                        )
                        obs, rew, done, _ = env.step(
                            action, [i == (args.episodes_per_trial - 1)]
                        )
                        # The reward we report is from the final episode in the trial
                        all_episodes_rew[i] += rew
                        if i == (args.episodes_per_trial - 1):
                            if success_support and env_base.is_success():
                                success = True
                            if done:
                                mask = np.asarray([True])
                        if done:
                            break

                with output_lock:
                    with open(
                        os.path.join(args.outdir, "evaluation.json"), "a"
                    ) as results_file:
                        results_file.write(
                            json.dumps(
                                {
                                    # For logging-sake, track the reward for each episode in the trial
                                    "episode_rewards": all_episodes_rew,
                                    # The 'reward' counted is still the reward of the final episode
                                    "reward": all_episodes_rew[
                                        args.episodes_per_trial - 1
                                    ],
                                    "success": success if success_support else "N/A",
                                    "environment": env_base.parameters,
                                    "model": args.load,
                                },
                                cls=NumpyEncoder,
                            )
                        )
                        results_file.write("\n")

    misc.async.run_async(args.eval_n_parallel, evaluator)


if __name__ == "__main__":
    main()
