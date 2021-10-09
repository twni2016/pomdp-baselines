# -*- coding: future_fstrings -*-
import sys, os, time

t0 = time.time()
import socket
import numpy as np
from ruamel.yaml import YAML
from utils import logger, system

from MRPO.examples.baselines import bench
from MRPO.examples.MRPO import base
from MRPO.examples.MRPO import MRPO_ppo2


def train_MRPO(
    env_id,
    total_episodes,
    seed,
    lr,
    paths,
    algorithm,
    policy,
    ncpu,
    nsteps,
    nminibatches,
):

    from MRPO.examples.baselines.common import set_global_seeds
    from MRPO.examples.baselines.common.vec_env.vec_normalize import VecNormalize
    import tensorflow as tf
    from MRPO.examples.baselines.common.vec_env.dummy_vec_env import DummyVecEnv

    # Set up environment
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=ncpu,
        inter_op_parallelism_threads=ncpu,
    )
    tf.Session(config=config).__enter__()

    assert ncpu == 1

    def make_env():
        env = base.make_env(env_id, outdir=logger.get_dir())
        # Set the env seed
        # (was missing before and is necessary to reproduce runs)
        # Other RNGs handled with 'set_global_seeds'
        env.seed(seed)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)
    set_global_seeds(seed)

    assert policy == "mlp"
    policy_fn = base.mlp_policy

    assert algorithm == "MRPO"
    print("Running MRPO with mujoco/roboschool settings")

    MRPO_ppo2.learn(
        # PPO2 mujoco defaults
        policy=policy_fn,
        env=env,
        env_id=env_id,
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=v["eval"]["log_interval"],
        ent_coef=0.0,
        lr=lr,
        cliprange=0.2,
        # total_timesteps=num_timesteps,
        total_episodes=total_episodes,
        # EPOpt specific
        paths=paths,
        # functions the same as old pposgd checkpointer
        save_interval=v["eval"]["save_interval"],
        eps_start=1.0,
        eps_end=40,
        eps_raise=1.005,
    )


yaml = YAML()
v = yaml.load(open(sys.argv[1]))

# system: device, threads, seed, pid
seed = v["seed"]
np.set_printoptions(precision=3, suppress=True)
pid = str(os.getpid())
if "SLURM_JOB_ID" in os.environ:
    pid += "_" + str(os.environ["SLURM_JOB_ID"])  # use job id

# set gpu
assert v["cuda"] == -1  # use tf cpu

# logs
exp_id = "logs/"
# exp_id = 'debug/'

env_type = v["env"]["env_type"]
env_name = v["env"]["env_name"]
exp_id += f"{env_type}/{env_name}/"
exp_id += "MRPO/"

os.makedirs(exp_id, exist_ok=True)
log_folder = os.path.join(exp_id, system.now_str())
logger_formats = ["stdout", "log", "csv"]
if v["eval"]["log_tensorboard"]:
    logger_formats.append("tensorboard")
logger.configure(dir=log_folder, format_strs=logger_formats, precision=4)
logger.log(f"preload cost {time.time() - t0:.2f}s")

os.system(f"cp -r MRPO/ {log_folder}")
os.system(f"cp {sys.argv[1]} {log_folder}/variant_{pid}.yml")
logger.log(sys.argv[1])
logger.log("pid", pid, socket.gethostname())
os.makedirs(os.path.join(logger.get_dir(), "save"))


train_MRPO(
    env_name,
    total_episodes=v["train"]["num_iters"],
    seed=seed,
    lr=3e-4,
    paths=100,  # number of trajectories to sample from each iteration
    algorithm="MRPO",
    policy="mlp",
    ncpu=1,
    nsteps=2048,
    nminibatches=32,
)
