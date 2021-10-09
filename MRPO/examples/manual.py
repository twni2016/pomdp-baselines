import argparse

import gym
from gym.utils.play import play

import sunblaze_envs


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--environment", type=str, default="SunblazeBreakout-v0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    episode = {"reward": 0, "initial": True}
    env = gym.make(args.environment)
    env.seed(args.seed)

    def reporter(obs_t, obs_tp1, action, reward, done, info):
        if episode["initial"]:
            episode["initial"] = False
            print("Environment parameters:")
            for key in sorted(env.unwrapped.parameters.keys()):
                print("  {}: {}".format(key, env.unwrapped.parameters[key]))

        episode["reward"] += reward
        if reward != 0:
            print("Reward:", episode["reward"])

        if done:
            print("*** GAME OVER ***")
            episode["reward"] = 0
            episode["initial"] = True

    play(env, callback=reporter)


if __name__ == "__main__":
    main()
