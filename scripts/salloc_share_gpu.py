# -*- coding: future_fstrings -*-
"""
Run on compute node (notice the mem limit): use cases are gpu-shared experiments 
"""
import os, sys
import subprocess
from itertools import product, cycle
import torch
import time
from ruamel.yaml import YAML
import datetime
import dateutil.tz
from pathlib import Path

yaml = YAML()
yaml.default_flow_style = None  # https://stackoverflow.com/a/56939573/9072850
os.environ["PYTHONPATH"] = os.getcwd()  # export PYTHONPATH=${PWD}:$PYTHONPATH
err_f = open("logs/error.log", "a")


def run(v, program):
    # v is a dictionary
    # program is program name
    def now_str():
        # use milisec to avoid conflict
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        return now.strftime("%m-%d:%H-%M:%S.%f")[:-4]

    os.makedirs("scripts/tmp_configs", exist_ok=True)
    tmp_config_name = os.path.join("scripts/tmp_configs", now_str() + ".yml")

    yaml.dump(v, Path(tmp_config_name))

    command = f"python3 {program} --cfg {tmp_config_name}"

    with open(os.devnull, "w") as f:
        subprocess.Popen(command, shell=True, stdout=f, stderr=err_f)

    return command


configs = [
    # ("configs/credit/catch/rnn.yml", "rnn"),
    # ("configs/credit/keytodoor/SR/rnn.yml", "rnn"),
    # ("configs/credit/keytodoor/LowVar/rnn.yml", "rnn"),
    # ("configs/credit/keytodoor/HighVar/rnn.yml", "rnn"),

    ("configs/credit/pendulum/rnn.yml", "rnn"),
    # ("configs/credit/ant/rnn.yml", "rnn"),
    # ("configs/credit/halfcheetah/rnn.yml", "rnn"),
    # ("configs/credit/hopper/rnn.yml", "rnn"),
    # ("configs/credit/walker/rnn.yml", "rnn"),
]

programs = {
    "mlp": "policies/main.py",
    "rnn": "policies/main.py",
}


seeds = [
    # 11,
    # 13,
    # 15,
    # 17,
    19,
    21,
    23,
    25,
]

num_gpus = torch.cuda.device_count()
print("num_gpus", num_gpus)
cuda_ids = cycle(list(range(num_gpus)) if num_gpus > 0 else [-1])

algos = [
    "sac",
    "td3",
    # "sacd",
]

gammas = [
    0.99,
    # 0.999,
]

entropies = [
    # 1.0,
    # 0.3,
    # 0.1,
    # 0.03,
    # 0.01,
    # 0.003,
]

for idx, (config, seed, algo, gamma) in enumerate(product(configs, seeds, algos, gammas)):
# for idx, (config, seed, entropy) in enumerate(product(configs, seeds, entropies)):

    config, code = config
    program = programs[code]
    v = yaml.load(open(config))

    v["seed"] = seed
    v["cuda"] = next(cuda_ids)  # NOTE: -1

    v["train"]["sampled_seq_len"] = -1
    v["train"]["num_updates_per_iter"] = 0.1
    v["policy"]["algo"] = algo
    v["policy"]["gamma"] = gamma

    # v["policy"]["automatic_entropy_tuning"] = False
    # v["policy"]["entropy_alpha"] = entropy
    # v["policy"]["target_entropy"] = entropy

    command = run(v, program)
    print(idx, seed, config, command)
    time.sleep(30)
