"""run_experiment.
Usage:
  run_experiment_saclstm.py run [--env=<kn>] [--steps=<kn>] [--seed=<kn>] [--render]
  run_experiment_saclstm.py (-h | --help)
Options:
  -h --help     Show this screen.
  --env=<kn>  Environment (see readme.txt) [default: PendulumV].
  --steps=<kn>  How many steps to run [default: 50000].
  --seed=<kn>  Random seed [default: 0].
"""

from docopt import docopt
import numpy as np
import gym
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import time, os, argparse, warnings
import scipy.io as sio
from copy import deepcopy
from network import SACRNN

arguments = docopt(__doc__, version="1.0")


def test_performance(agent_test, env_test, action_filter, times=5):

    EpiTestRet = 0

    for _ in range(times):

        # reset each episode
        s0 = env_test.reset()
        a = agent_test.init_episode(s0)

        for _ in range(max_steps):
            if np.any(np.isnan(a)):
                raise ValueError
            sp, r, done, _ = env_test.step(action_filter(a))
            a = agent_test.select(
                sp, action_return="normal"
            )  # use tanh(mu_a) for evaluating performance
            EpiTestRet += r
            if done:
                break

    EpiTestRet_mean = 0

    for _ in range(times):

        # reset each episode
        s0 = env_test.reset()
        a = agent_test.init_episode(s0)

        for _ in range(max_steps):
            if np.any(np.isnan(a)):
                raise ValueError
            sp, r, done, _ = env_test.step(action_filter(a))
            a = agent_test.select(
                sp, action_return="mean"
            )  # use tanh(mu_a) for evaluating performance
            EpiTestRet_mean += r
            if done:
                break

    return EpiTestRet / times, EpiTestRet_mean / times


savepath = "./data/"

if os.path.exists(savepath):
    warnings.warn("{} exists (possibly so do data).".format(savepath))
else:
    os.makedirs(savepath)

seed = int(arguments["--seed"])  # random seed
np.random.seed(seed)
torch.manual_seed(seed)

beta_h = "auto_1.0"
batch_size = 256
reward_scale = 1.0
grad_clip = False
computation_mode = "implicit"

optimizer_st = "adam"
minibatch_size = 4
seq_len = 64

step_start_rl = 1000

train_step_rl = 1  # how many times of RL training after step_start_rl
train_freq_rl = 1.0 / train_step_rl

max_all_steps = int(arguments["--steps"])  # total steps to learn
step_perf_eval = 2000  # how many steps to do evaluation

env_name = arguments["--env"]

# Task-specific

if env_name == "Sequential":

    from task import TaskT

    env = TaskT(3)
    env_test = TaskT(3)
    action_filter = lambda a: a.reshape([-1])

    max_steps = 128
    est_min_steps = 10

elif env_name == "CartPole":

    from task import ContinuousCartPoleEnv

    env = ContinuousCartPoleEnv()
    env_test = ContinuousCartPoleEnv()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 5

elif env_name == "CartPoleP":

    from task import CartPoleP

    env = CartPoleP()
    env_test = CartPoleP()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 10

elif env_name == "CartPoleV":

    from task import CartPoleV

    env = CartPoleV()
    env_test = CartPoleV()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 10

elif env_name == "Pendulum":

    import gym

    env = gym.make("Pendulum-v0")
    env_test = gym.make("Pendulum-v0")

    action_filter = (
        lambda a: a.reshape([-1]) * 2
    )  # because range of pendulum's action is [-2, 2]. For other environments, * 2 is not needed

    max_steps = 200
    est_min_steps = 199

elif env_name == "PendulumP":

    from task import PendulumP

    env = PendulumP()
    env_test = PendulumP()

    action_filter = lambda a: a.reshape([-1]) * 2

    max_steps = 200
    est_min_steps = 199

elif env_name == "PendulumV":

    from task import PendulumV

    env = PendulumV()
    env_test = PendulumV()

    action_filter = lambda a: a.reshape([-1]) * 2

    max_steps = 200
    est_min_steps = 199

elif env_name == "Hopper":

    import gym
    import roboschool

    env = gym.make("RoboschoolHopper-v1")
    env_test = gym.make("RoboschoolHopper-v1")

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 5

elif env_name == "HopperP":

    from task import RsHopperP

    env = RsHopperP()
    env_test = RsHopperP()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 5

elif env_name == "HopperV":

    from task import RsHopperV

    env = RsHopperV()
    env_test = RsHopperV()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 5

elif env_name == "Walker2d":

    import gym
    import roboschool

    env = gym.make("RoboschoolWalker2d-v1")
    env_test = gym.make("RoboschoolWalker2d-v1")

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 5

elif env_name == "Walker2dV":

    from task import RsWalker2dV

    env = RsWalker2dV()
    env_test = RsWalker2dV()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 5

elif env_name == "Walker2dP":

    from task import RsWalker2dP

    env = RsWalker2dP()
    env_test = RsWalker2dP()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 5

elif env_name == "Ant":

    import gym
    import roboschool

    env = gym.make("RoboschoolAnt-v1")
    env_test = gym.make("RoboschoolAnt-v1")

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 20

elif env_name == "AntV":

    from task import RsAntV

    env = RsAntV()
    env_test = RsAntV()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 20

elif env_name == "AntP":

    from task import RsAntP

    env = RsAntP()
    env_test = RsAntP()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 20

# ----------------initialize-------------
max_episodes = int(max_all_steps / est_min_steps) + 1  # for replay buffer

agent = SACRNN(
    env.observation_space.shape[0],
    env.action_space.shape[0],
    gamma=0.99,
    reward_scale=reward_scale,
    beta_h=beta_h,
)
agent_test = SACRNN(
    env.observation_space.shape[0],
    env.action_space.shape[0],
    gamma=0.99,
    reward_scale=reward_scale,
    beta_h=beta_h,
)

# SP_real = np.zeros([max_episodes, max_steps, env.observation_space.shape[0]], dtype=np.float32)
S_real = np.zeros(
    [max_episodes, max_steps + 1, env.observation_space.shape[0]], dtype=np.float32
)
A_real = np.zeros(
    [max_episodes, max_steps, env.action_space.shape[0]], dtype=np.float32
)
R_real = np.zeros([max_episodes, max_steps], dtype=np.float32)
D_real = np.zeros([max_episodes, max_steps], dtype=np.float32)  # done
V_real = np.zeros(
    [max_episodes, max_steps], dtype=np.float32
)  # whether a step is value: 1 (compute gradient at this step) or 0 (stop gradient at this step)

performance_wrt_step = []
performance_mean_action_wrt_step = []

global_steps = []

e_real = 0
global_step = 0

t_just = 0

while global_step < max_all_steps:

    s = env.reset()
    a = agent.init_episode(s)

    for t in range(max_steps):

        if global_step == max_all_steps:
            break

        sp, r, done, _ = env.step(action_filter(a))

        A_real[e_real, t, :] = a
        S_real[e_real, t, :] = s.reshape([-1])
        S_real[e_real, t + 1, :] = sp.reshape([-1])
        R_real[e_real, t] = r
        D_real[e_real, t] = 1 if done else 0
        V_real[e_real, t] = 1

        a = agent.select(sp)

        global_step += 1
        s = deepcopy(sp)

        if global_step == step_start_rl + 1:  # pretrain
            for _ in range(5000):

                weights = np.sum(V_real[:e_real], axis=-1) + 2 * seq_len - 2
                sample_es = np.random.choice(
                    e_real, minibatch_size, p=weights / weights.sum()
                )

                S = S_real[sample_es, :-1, :].reshape(
                    [minibatch_size, -1, env.observation_space.shape[0]]
                )
                SP = S_real[sample_es, 1:, :].reshape(
                    [minibatch_size, -1, env.observation_space.shape[0]]
                )
                A = A_real[sample_es].reshape(
                    [minibatch_size, -1, env.action_space.shape[0]]
                )
                R = R_real[sample_es].reshape([minibatch_size, -1, 1])
                D = D_real[sample_es].reshape([minibatch_size, -1, 1])
                V = V_real[sample_es].reshape([minibatch_size, -1, 1])

                agent.learn(
                    S,
                    SP,
                    R,
                    A,
                    D,
                    V,
                    computation=computation_mode,
                    grad_clip=grad_clip,
                    seq_len=seq_len,
                )

        if global_step > step_start_rl and np.random.rand() < train_freq_rl:

            for _ in range(max(1, int(train_freq_rl))):

                weights = np.sum(V_real[:e_real], axis=-1) + 2 * seq_len - 2
                sample_es = np.random.choice(
                    e_real, minibatch_size, p=weights / weights.sum()
                )

                S = S_real[sample_es, :-1, :].reshape(
                    [minibatch_size, -1, env.observation_space.shape[0]]
                )
                SP = S_real[sample_es, 1:, :].reshape(
                    [minibatch_size, -1, env.observation_space.shape[0]]
                )
                A = A_real[sample_es].reshape(
                    [minibatch_size, -1, env.action_space.shape[0]]
                )
                R = R_real[sample_es].reshape([minibatch_size, -1, 1])
                D = D_real[sample_es].reshape([minibatch_size, -1, 1])
                V = V_real[sample_es].reshape([minibatch_size, -1, 1])

                agent.learn(
                    S,
                    SP,
                    R,
                    A,
                    D,
                    V,
                    computation=computation_mode,
                    grad_clip=grad_clip,
                    seq_len=seq_len,
                )

        if global_step % step_perf_eval == 0:
            agent_test.load_state_dict(agent.state_dict())  # update agent_test
            EpiTestRet, EpiTestRet_mean = test_performance(
                agent_test, env_test, action_filter, times=5
            )
            performance_wrt_step.append(EpiTestRet)
            performance_mean_action_wrt_step.append(EpiTestRet_mean)
            global_steps.append(global_step)
            warnings.warn(
                env_name
                + ": global step: {}, : steps {}, test return {}".format(
                    global_step, t, EpiTestRet
                )
            )

        if done:
            break

    print(
        env_name
        + " -- episode {} : steps {}, mean reward {}".format(
            e_real, t, np.mean(R_real[e_real])
        )
    )
    e_real += 1

performance_wrt_step = np.reshape(performance_wrt_step, [-1]).astype(np.float64)
performance_mean_action_wrt_step_array = np.reshape(
    performance_mean_action_wrt_step, [-1]
).astype(np.float64)
global_steps = np.reshape(global_steps, [-1]).astype(np.float64)

data = {
    "seq_len": seq_len,
    "max_steps": max_steps,
    "max_episodes": max_episodes,
    "step_start_rl": step_start_rl,
    "reward_scale": reward_scale,
    "beta_h": beta_h,
    "minibatch_size": minibatch_size,
    "train_step_rl": train_step_rl,
    "R": np.sum(R_real, axis=-1).astype(np.float64),
    "steps": np.sum(V_real, axis=-1).astype(np.float64),
    "performance_wrt_step": performance_wrt_step,
    "performance_mean_action_wrt_step": performance_mean_action_wrt_step_array,
    "global_steps": global_steps,
}

sio.savemat(savepath + env_name + "_saclstm.mat", data)
torch.save(agent, savepath + env_name + "_saclstm.model")
