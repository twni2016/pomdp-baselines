"""run_experiment.
Usage:
  run_experiment_slac.py run [--env=<kn>] [--steps=<kn>] [--seed=<kn>] [--render]
  run_experiment_slac.py (-h | --help)
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
import torch.nn as nn
from torch.autograd import Variable
import time, os, argparse, warnings
import scipy.io as sio
from copy import deepcopy
from slac import SLAC

arguments = docopt(__doc__, version="1.0")


def test_performance(agent_test, env_test, action_filter, times=5):

    EpiTestRet = 0

    for _ in range(times):

        # reset each episode
        sp_seq = np.zeros([seq_len, env.observation_space.shape[0] + 1])
        s = env_test.reset()
        sp_seq[-1, :-1] = s
        sp_seq[-1, -1] = 0.0  # reward padding

        a = agent.select(sp_seq)
        for _ in range(max_steps):
            if np.any(np.isnan(a)):
                raise ValueError
            sp, r, done, _ = env_test.step(action_filter(a))
            sp_seq[:-1] = deepcopy(sp_seq[1:])
            sp_seq[-1, :-1] = deepcopy(sp)
            sp_seq[-1, -1] = r
            a = agent_test.select(
                sp_seq, action_return="normal"
            )  # use tanh(mu_a) for evaluating performance
            EpiTestRet += r
            if done:
                break

    EpiTestRet_mean = 0

    for _ in range(times):

        # reset each episode
        sp_seq = np.zeros([seq_len, env.observation_space.shape[0] + 1])
        s = env_test.reset()
        sp_seq[-1, :-1] = s
        sp_seq[-1, -1] = 0.0  # reward padding

        a = agent.select(sp_seq)
        for _ in range(max_steps):
            if np.any(np.isnan(a)):
                raise ValueError
            sp, r, done, _ = env_test.step(action_filter(a))
            sp_seq[:-1] = deepcopy(sp_seq[1:])
            sp_seq[-1, :-1] = deepcopy(sp)
            sp_seq[-1, -1] = r
            a = agent_test.select(
                sp_seq, action_return="mean"
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

# Shared
computation_mode = "implicit"
beta_h = "auto_1.0"
optimizer = "adam"
batch_size = 32
seq_len = 8
reward_scale = 1.0
action_feedback = True
grad_clip = False
gamma = 0.99

equal_pad = True
pre_train = True
nc = False
model_act_fn = nn.Tanh
sigx = "auto"

max_all_steps = int(arguments["--steps"])  # total steps to learn
step_perf_eval = 2000  # how many steps to do evaluation

env_name = arguments["--env"]

step_start_rl = 1000
step_start_st = 1000

train_step_rl = 1
train_freq_rl = 1.0 / train_step_rl

train_step_st = 1
train_freq_st = 1.0 / train_step_st

if arguments["--render"]:
    rendering = True
else:
    rendering = False

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

agent = SLAC(
    input_size=env.observation_space.shape[0] + 1,
    action_size=env.action_space.shape[0],
    seq_len=seq_len,
    beta_h=beta_h,
    model_act_fn=model_act_fn,
    sigx=sigx,
)

agent_test = SLAC(
    input_size=env.observation_space.shape[0] + 1,
    action_size=env.action_space.shape[0],
    seq_len=seq_len,
    beta_h=beta_h,
    model_act_fn=model_act_fn,
    sigx=sigx,
)

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
)  # whether a step is valid value: 1 (compute gradient at this step) or 0 (stop gradient at this step)

performance_wrt_step = []
performance_mean_action_wrt_step = []
global_steps = []

e_real = 0
global_step = 0
t_just = 0

while global_step < max_all_steps:

    sp_seq = np.zeros([seq_len, env.observation_space.shape[0] + 1])
    s = env.reset()
    S_real[e_real, 0] = s.reshape([-1])
    sp_seq[-1, :-1] = s
    sp_seq[-1, -1] = 0.0
    if equal_pad:
        for tau in range(seq_len - 1):
            sp_seq[tau, :-1] = s

    a = agent.select(sp_seq)

    for t in range(max_steps):

        if global_step == max_all_steps:
            break

        sp, r, done, _ = env.step(action_filter(a))

        sp_seq[:-1] = deepcopy(sp_seq[1:])
        sp_seq[-1, :-1] = deepcopy(sp)
        sp_seq[-1, -1] = r

        A_real[e_real, t] = a
        S_real[e_real, t + 1] = sp.reshape([-1])
        R_real[e_real, t] = r
        D_real[e_real, t] = 1 if done else 0
        V_real[e_real, t] = 1

        a = agent.select(sp_seq)

        global_step += 1
        s = deepcopy(sp)

        if pre_train and global_step == step_start_st + 1:
            for _ in range(5000):
                weights = np.sum(V_real[:e_real], axis=-1) + 2 * seq_len - 2
                sample_es = np.random.choice(
                    e_real, batch_size, p=weights / weights.sum()
                )

                SP = S_real[sample_es, 1:].reshape(
                    [batch_size, -1, env.observation_space.shape[0]]
                )
                A = A_real[sample_es].reshape(
                    [batch_size, -1, env.action_space.shape[0]]
                )
                R = R_real[sample_es].reshape([batch_size, -1, 1])
                V = V_real[sample_es].reshape([batch_size, -1, 1])

                agent.train_st(
                    x_obs=np.concatenate((SP, R), axis=-1), a_obs=A, r_obs=R, validity=V
                )

        if global_step > step_start_st and np.random.rand() < train_freq_st:

            for _ in range(max(1, int(train_freq_st))):
                weights = np.sum(V_real[:e_real], axis=-1) + 2 * seq_len - 2
                sample_es = np.random.choice(
                    e_real, batch_size, p=weights / weights.sum()
                )

                SP = S_real[sample_es, 1:].reshape(
                    [batch_size, -1, env.observation_space.shape[0]]
                )
                A = A_real[sample_es].reshape(
                    [batch_size, -1, env.action_space.shape[0]]
                )
                R = R_real[sample_es].reshape([batch_size, -1, 1])
                V = V_real[sample_es].reshape([batch_size, -1, 1])

                agent.train_st(
                    x_obs=np.concatenate((SP, R), axis=-1), a_obs=A, r_obs=R, validity=V
                )

        if global_step > step_start_rl and np.random.rand() < train_freq_rl:

            for _ in range(max(1, int(train_freq_rl))):
                weights = np.sum(V_real[:e_real], axis=-1) + 2 * seq_len - 2
                sample_es = np.random.choice(
                    e_real, batch_size, p=weights / weights.sum()
                )

                SP = S_real[sample_es, 1:].reshape(
                    [batch_size, -1, env.observation_space.shape[0]]
                )
                S0 = S_real[sample_es, 0].reshape(
                    [batch_size, env.observation_space.shape[0]]
                )
                A = A_real[sample_es].reshape(
                    [batch_size, -1, env.action_space.shape[0]]
                )
                R = R_real[sample_es].reshape([batch_size, -1, 1])
                D = D_real[sample_es].reshape([batch_size, -1, 1])
                V = V_real[sample_es].reshape([batch_size, -1, 1])

                agent.train_rl_sac(
                    x_obs=np.concatenate((SP, R), axis=-1),
                    s_0=S0,
                    a_obs=A,
                    r_obs=R,
                    d_obs=D,
                    validity=V,
                    gamma=0.99,
                    equal_pad=equal_pad,
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
    "sigx": sigx,
    "beta_h": beta_h,
    "gamma": gamma,
    "max_steps": max_steps,
    "max_episodes": max_episodes,
    "step_start_st": step_start_st,
    "step_start_rl": step_start_rl,
    "batch_size": batch_size,
    "train_step_rl": train_step_rl,
    "train_step_st": train_step_st,
    "R": np.sum(R_real, axis=-1).astype(np.float64),
    "steps": np.sum(V_real, axis=-1).astype(np.float64),
    "performance_wrt_step": performance_wrt_step,
    "performance_mean_action_wrt_step": performance_mean_action_wrt_step_array,
    "global_steps": global_steps,
}

sio.savemat(savepath + env_name + "_" + "slac" + ".mat", data)
torch.save(agent, savepath + env_name + "_" + "slac" + ".model")
