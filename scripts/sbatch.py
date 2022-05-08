# -*- coding: future_fstrings -*-
"""
Run on login node
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


def get_sbatch_command(
    time_limit="24:00:00", mem="10G", n_cpus=1, gpu="volta"
):
    cmd = ["sbatch"]
    cmd.extend(["-o", "/dev/null"])
    cmd.extend(["-e", "logs/error-%j.out"])
    cmd.extend(["-t", time_limit])
    cmd.extend(["-c", str(n_cpus)])
    cmd.extend(["--mem", mem])
    cmd.extend(["--gres", f"gpu:{gpu}:1"])

    cmd = " ".join(cmd)
    return cmd

'''
GPU list:
- volta (v100)
- 2080Ti


'''

sbatch_cmd = get_sbatch_command(
    time_limit="72:00:00", mem="5G", n_cpus=1, gpu="2080Ti",
)

def get_python_cmd(v, program):
    # v is a dictionary
    # program is program name
    def now_str():
        # use milisec to avoid conflict
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        return now.strftime("%m-%d:%H-%M:%S.%f")[:-4]

    os.makedirs("scripts/tmp_configs", exist_ok=True)
    tmp_config_name = os.path.join("scripts/tmp_configs", now_str() + ".yml")

    yaml.dump(v, Path(tmp_config_name))

    return program, tmp_config_name


yaml = YAML()
yaml.default_flow_style = None  # https://stackoverflow.com/a/56939573/9072850

configs = [
    # ("configs/credit/catch/rnn.yml", "rnn"),
    # ("configs/credit/keytodoor/SR/rnn.yml", "rnn"),
    # ("configs/credit/keytodoor/LowVar/rnn.yml", "rnn"),
    # ("configs/credit/keytodoor/HighVar/rnn.yml", "rnn"),
    # ("configs/credit/pendulum/rnn.yml", "rnn"),
    ("configs/credit/ant/rnn.yml", "rnn"),
    ("configs/credit/halfcheetah/rnn.yml", "rnn"),
    ("configs/credit/hopper/rnn.yml", "rnn"),
    ("configs/credit/walker/rnn.yml", "rnn"),
]

programs = {
    "mlp": "policies/main.py",
    "rnn": "policies/main.py",
}


seeds = [
    11,
    # 13,
    # 15,
    # 17,
    # 19,
    # 21,
    # 23,
    # 25,
]

algos = [
    "sac",
    "td3",
    # "sacd",
]

gammas = [
    # 0.99,
    0.999,
]

entropies = [
    # 1.0,
    # 0.3,
    # 0.1,
    # 0.03,
    # 0.01,
    # 0.003,
]

for idx, (config, seed, algo, gamma) in enumerate(
    product(configs, seeds, algos, gammas)
):
    # for idx, (config, seed, entropy) in enumerate(product(configs, seeds, entropies)):

    config, code = config
    program = programs[code]
    v = yaml.load(open(config))

    v["seed"] = seed
    v["cuda"] = 0  # NOTE

    v["train"]["sampled_seq_len"] = -1
    v["train"]["num_updates_per_iter"] = 0.25
    v["policy"]["algo"] = algo
    v["policy"]["gamma"] = gamma

    # v["policy"]["automatic_entropy_tuning"] = False
    # v["policy"]["entropy_alpha"] = entropy
    # v["policy"]["target_entropy"] = entropy


    program, tmp_config_name = get_python_cmd(v, program)

    cmd = f"{sbatch_cmd} run {program} {tmp_config_name}"
    print(cmd)
    os.system(cmd)

    time.sleep(30)