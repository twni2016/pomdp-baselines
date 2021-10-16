import sys, os, time

t0 = time.time()
import socket
import numpy as np
import torch
from ruamel.yaml import YAML
from utils import system, logger

from torchkit.pytorch_utils import set_gpu_mode
from BOReL.metalearner import MetaLearner

yaml = YAML()
v = yaml.load(open(sys.argv[1]))

# system: device, threads, seed, pid
seed = v["seed"]
torch.set_num_threads(1)
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)
system.reproduce(seed)
pid = str(os.getpid())
if "SLURM_JOB_ID" in os.environ:
    pid += "_" + str(os.environ["SLURM_JOB_ID"])  # use job id

# set gpu
set_gpu_mode(torch.cuda.is_available() and v["cuda"] >= 0, v["cuda"])

# logs
env_type = v["env"]["env_type"]
if len(v["env"]["env_name"].split("-")) == 3:
    # pomdp env: name-{F/P/V}-v0
    env_name, env_type, _ = v["env"]["env_name"].split("-")
    env_name = env_name + "/" + env_type
else:
    env_name = v["env"]["env_name"]
exp_id = f"logs/{env_type}/{env_name}/"

exp_id += "varibad/"
# exp_id = 'debug'

os.makedirs(exp_id, exist_ok=True)
log_folder = os.path.join(exp_id, system.now_str())
logger_formats = ["stdout", "log", "csv"]
if v["train"]["log_tensorboard"]:
    logger_formats.append("tensorboard")
logger.configure(dir=log_folder, format_strs=logger_formats, precision=4)
logger.log(f"preload cost {time.time() - t0:.2f}s")

os.system(f"cp -r BOReL/ {log_folder}")
os.system(f"cp {sys.argv[1]} {log_folder}/variant_{pid}.yml")
logger.log(sys.argv[1])
logger.log("pid", pid, socket.gethostname())
if v["train"]["save_interval"] > 0:
    os.makedirs(os.path.join(logger.get_dir(), "save"))


# start training
learner = MetaLearner(
    env_args=v["env"],
    train_args=v["train"],
    policy_args=v["policy"],
    vae_args=v["vae"],
    seed=seed,
)

learner.train()
