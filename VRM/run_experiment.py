"""
Adapt the code from https://github.com/oist-cnru/Variational-Recurrent-Models
into our style (running command, environments, training protocol, logging)
"""
import sys, os, time

t0 = time.time()
import socket
from ruamel.yaml import YAML
from utils import system, logger
from torchkit import pytorch_utils as ptu

import gym
import numpy as np
import torch
import envs
from VRM.vrdm import VRM, VRDM

yaml = YAML()
v = yaml.load(open(sys.argv[1]))

seed = v["seed"]  # random seed
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
exp_id += "VRM/"

os.makedirs(exp_id, exist_ok=True)
log_folder = os.path.join(exp_id, system.now_str())
logger_formats = ["stdout", "log", "csv"]
if v["eval"]["log_tensorboard"]:
    logger_formats.append("tensorboard")
logger.configure(dir=log_folder, format_strs=logger_formats, precision=4)
logger.log(f"preload cost {time.time() - t0:.2f}s")

os.system(f"cp -r VRM/ {log_folder}")
os.system(f"cp {sys.argv[1]} {log_folder}/variant_{pid}.yml")
logger.log(sys.argv[1])
logger.log("pid", pid, socket.gethostname())
os.makedirs(os.path.join(logger.get_dir(), "save"))


env_name = v["env"]["env_name"]
# assert "BLT" in env_name # pybullet environments
env = gym.make(env_name)
env_test = gym.make(env_name)
action_filter = lambda a: a.reshape([-1])
max_steps = env._max_episode_steps
# est_min_steps = 5

beta_h = "auto"
optimizer_st = "adam"
minibatch_size = 4
seq_len = 64
lr_vrm = 8e-4
gamma = 0.99
max_all_steps = v["train"]["num_iters"] * max_steps  # total steps to learn
step_perf_eval = (
    v["eval"]["log_interval"] * max_steps
)  # how many steps to do evaluation
save_interval = v["eval"]["save_interval"] * max_steps

rnn_type = "mtlstm"
d_layers = [
    256,
]
z_layers = [
    64,
]
x_phi_layers = [128]
decode_layers = [128, 128]

value_layers = [256, 256]
policy_layers = [256, 256]

step_start_rl = 1000
step_start_st = 1000
step_end_st = np.inf
fim_train_times = 5000  # NOTE: a bit slow

train_step_rl = 1  # how many times of RL training after step_start_rl
train_step_st = 5

train_freq_rl = 1.0 / train_step_rl
train_freq_st = 1.0 / train_step_st

max_episodes = int(v["train"]["buffer_size"])  # for replay buffer
# max_episodes = int(v['train']['buffer_size_ratio'] * max_all_steps / est_min_steps)
# NOTE: we have to reduce the buffer size because it is too large to be in the RAM for multiple experiments
# with 0.05 ratio, it already costs around 2G, and can store 15000 episodes >> 1500 episodes
# thus it should not affect the method (not exceeding the max_episode)
logger.log("max_episodes in the buffer:", max_episodes)

fim = VRM(
    input_size=env.observation_space.shape[0] + 1,
    action_size=env.action_space.shape[0],
    rnn_type=rnn_type,
    d_layers=d_layers,
    z_layers=z_layers,
    decode_layers=decode_layers,
    x_phi_layers=x_phi_layers,
    optimizer=optimizer_st,
    lr_st=lr_vrm,
).to(ptu.device)

klm = VRM(
    input_size=env.observation_space.shape[0] + 1,
    action_size=env.action_space.shape[0],
    rnn_type=rnn_type,
    d_layers=d_layers,
    z_layers=z_layers,
    decode_layers=decode_layers,
    x_phi_layers=x_phi_layers,
    optimizer=optimizer_st,
    lr_st=lr_vrm,
).to(ptu.device)

agent = VRDM(
    fim,
    klm,
    gamma=gamma,
    beta_h=beta_h,
    value_layers=value_layers,
    policy_layers=policy_layers,
).to(ptu.device)
logger.log(agent)
agent_test = VRDM(
    fim,
    klm,
    gamma=gamma,
    beta_h=beta_h,
    value_layers=value_layers,
    policy_layers=policy_layers,
).to(ptu.device)


SP_real = np.zeros(
    [max_episodes, max_steps, env.observation_space.shape[0]], dtype=np.float32
)  # observation (t+1)
A_real = np.zeros(
    [max_episodes, max_steps, env.action_space.shape[0]], dtype=np.float32
)  # action
R_real = np.zeros([max_episodes, max_steps], dtype=np.float32)  # reward
D_real = np.zeros([max_episodes, max_steps], dtype=np.float32)  # done
V_real = np.zeros(
    [max_episodes, max_steps], dtype=np.float32
)  # mask, indicating whether a step is valid. value: 1 (compute gradient at this step) or 0 (stop gradient at this step)

performance_wrt_step = []
global_steps = []

e_real = 0
e_capacity = 0
global_step = 0


def test_performance(agent_test, env_test, action_filter, times=10):

    EpiTestRet = 0
    steps = 0
    for _ in range(times):

        # reset each episode
        s0 = env_test.reset().astype(np.float32)
        r0 = np.array([0.0], dtype=np.float32)
        x0 = np.concatenate([s0, r0])
        a = agent_test.init_episode(x0).reshape(-1)

        for _ in range(max_steps):
            if np.any(np.isnan(a)):
                raise ValueError
            sp, r, done, _ = env_test.step(action_filter(a))
            # NOTE: we change it to mean for fair comparison
            a = agent_test.select(sp, r, action_return="mean")
            EpiTestRet += r
            steps += 1
            if done:
                break

    return EpiTestRet / times, steps / times


def epsiode_candidates():
    indices = list(range(0, e_capacity))
    try:
        indices.remove(e_real)
    except ValueError:
        pass
    return indices


#  Run
loss_st, loss_critic, loss_actor = 0.0, 0.0, 0.0

while global_step < max_all_steps:

    s0 = env.reset().astype(np.float32)
    r0 = np.array([0.0], dtype=np.float32)
    x0 = np.concatenate([s0, r0])
    a = agent.init_episode(x0).reshape(-1)

    for t in range(max_steps):

        if global_step == max_all_steps:
            break

        if np.any(np.isnan(a)):
            raise ValueError

        sp, r, done, _ = env.step(action_filter(a))

        A_real[e_real, t, :] = a
        SP_real[e_real, t, :] = sp.reshape([-1])
        R_real[e_real, t] = r
        D_real[e_real, t] = 1 if done else 0
        V_real[e_real, t] = 1

        a = agent.select(sp, r)

        global_step += 1

        if global_step == step_start_st + 1:
            print("Start training the first-impression model!")
            sample = epsiode_candidates()
            _, _, loss_st = agent.learn_st(
                True,
                False,
                SP_real[sample],
                A_real[sample],
                R_real[sample],
                D_real[sample],
                V_real[sample],
                times=fim_train_times,
                minibatch_size=minibatch_size,
            )
            print("Finish training the first-impression model!")
            print("Start training the keep-learning model!")

        if (
            global_step > step_start_st
            and global_step <= step_end_st
            and np.random.rand() < train_freq_st
        ):
            # every 5 steps train once: train Keep-Learning Model
            t0 = time.time()
            sample = epsiode_candidates()
            _, _, loss_st = agent.learn_st(
                False,
                True,
                SP_real[sample],
                A_real[sample],
                R_real[sample],
                D_real[sample],
                V_real[sample],
                times=max(1, int(train_freq_st)),
                minibatch_size=minibatch_size,
            )
            print("KLM time", time.time() - t0)  # 0.84s (GPU) vs 0.60s (CPU)

        if global_step > step_start_rl and np.random.rand() < train_freq_rl:
            # every 1 step train once: train RL (SAC)
            t0 = time.time()
            sample = epsiode_candidates()
            if global_step == step_start_rl + 1:
                print("Start training the RL controller!")
            loss_v, loss_a, loss_q = agent.learn_rl_sac(
                SP_real[sample],
                A_real[sample],
                R_real[sample],
                D_real[sample],
                V_real[sample],
                times=max(1, int(train_freq_rl)),
                minibatch_size=minibatch_size,
                seq_len=seq_len,
            )
            loss_critic = loss_v + loss_q
            loss_actor = loss_a
            print("SAC time", time.time() - t0)  # 0.24s (GPU) vs 0.23s (CPU)

        if global_step % step_perf_eval == 0:
            agent_test.load_state_dict(agent.state_dict())  # update agent_test
            EpiTestRet, EpiTestStep = test_performance(
                agent_test, env_test, action_filter, times=v["env"]["num_eval_tasks"]
            )

            logger.record_step(global_step)
            logger.record_tabular("z/env_steps", global_step)
            logger.record_tabular("z/rollouts", e_capacity)
            logger.record_tabular("z/time_interval", time.time() - t0)
            logger.record_tabular("metrics/total_steps_eval", EpiTestStep)
            logger.record_tabular("metrics/return_eval_total", EpiTestRet)
            logger.dump_tabular()
            t0 = time.time()

            if global_step > 0.75 * max_all_steps and global_step % save_interval == 0:
                save_path = os.path.join(
                    logger.get_dir(),
                    "save",
                    f"agent_{global_step // max_steps}_perf{EpiTestRet:.3f}.pt",
                )
                torch.save(agent.state_dict(), save_path)

        if done:
            break

    # End of one episode
    print(
        env_name
        + " -- episode {} : steps {}, sum of running rewards {}".format(
            e_real, t, np.sum(R_real[e_real])
        )
    )
    e_real += 1
    e_real %= max_episodes  # added by ours to avoid error
    e_capacity = min(e_capacity + 1, max_episodes)
    if e_capacity == max_episodes:
        logger.log("WARNING: EXCEEDING THE MAX_EPSIODE IN THE BUFFER!!!")

    logger.record_step(global_step)
    logger.record_tabular("z/env_steps", global_step)
    logger.record_tabular("z/rollouts", e_capacity)
    logger.record_tabular("VRM/model_loss", loss_st)
    logger.record_tabular("VRM/loss_critic", loss_critic)
    logger.record_tabular("VRM/loss_actor", loss_actor)
    logger.dump_tabular()
