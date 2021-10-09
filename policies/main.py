# -*- coding: future_fstrings -*-
import sys, os, time

t0 = time.time()
import socket
import numpy as np
import torch
from ruamel.yaml import YAML
from utils import system, logger

from torchkit.pytorch_utils import set_gpu_mode
from policies.learner import Learner

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

if "AntSemiCircle" in env_name:
    if v["env"]["modify_init_state_dist"] == False:
        prefix = "fixed"
    elif v["env"]["on_circle_init_state"] == False:
        prefix = "exclude_sc"
    else:
        prefix = "uniform"
    exp_id += prefix + "/"

arch, algo = v["policy"]["arch"], v["policy"]["algo"]
assert arch in ["mlp", "lstm", "gru"]
assert algo in ["td3", "sac"]
exp_id += f"{algo}_{arch}"
if "separate" in v["policy"] and v["policy"]["separate"] == False:
    exp_id += "_shared"
exp_id += "/"

if arch in ["lstm", "gru"]:
    exp_id += f"len-{v['train']['sampled_seq_len']}/bs-{v['train']['batch_size']}/"
    exp_id += f"baseline-{v['train']['sample_weight_baseline']}/"
    exp_id += f"freq-{v['train']['num_updates_per_iter']}/"
    assert v["policy"]["state_embedding_size"] > 0
    policy_input_str = "o"
    if v["policy"]["action_embedding_size"] > 0:
        policy_input_str += "a"
    if v["policy"]["reward_embedding_size"] > 0:
        policy_input_str += "r"
    exp_id += policy_input_str + "/"

os.makedirs(exp_id, exist_ok=True)
log_folder = os.path.join(exp_id, system.now_str())
logger_formats = ["stdout", "log", "csv"]
if v["eval"]["log_tensorboard"]:
    logger_formats.append("tensorboard")
logger.configure(dir=log_folder, format_strs=logger_formats, precision=4)
logger.log(f"preload cost {time.time() - t0:.2f}s")

os.system(f"cp -r policies/ {log_folder}")
os.system(f"cp {sys.argv[1]} {log_folder}/variant_{pid}.yml")
logger.log(sys.argv[1])
logger.log("pid", pid, socket.gethostname())
os.makedirs(os.path.join(logger.get_dir(), "save"))


# start training
learner = Learner(
    env_args=v["env"],
    train_args=v["train"],
    eval_args=v["eval"],
    policy_args=v["policy"],
    seed=seed,
)

learner.train()
