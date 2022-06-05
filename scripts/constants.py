# tags
x_tag = ("z/env_steps", "env_steps")
y_tags = [  ### raw_atg, processed tag
    ## reward/succ tags
    ("metrics/return_eval_total", "return"),
    # ("metrics/return_train_total", "return_train"),
    # ("metrics/success_rate_eval", "success_rate"),
    ("metrics/return_eval_avg", "return_avg"),
    ("metrics/return_eval_worst", "return_worst"),
    ("metrics/succ_eval_DD", "succ_DD"),
    ("metrics/succ_eval_DR", "succ_DR"),
    ("metrics/succ_eval_DE", "succ_DE"),
    ("metrics/succ_eval_RD", "succ_RD"),
    ("metrics/succ_eval_RR", "succ_RR"),
    ("metrics/succ_eval_RE", "succ_RE"),
    ("metrics/succ_eval_EE", "succ_EE"),
]

generalization_tags = {
    "Default": ["succ_DD"],
    "Interpolation": ["succ_RR"],
    "Extrapolation": ["succ_DR", "succ_DE", "succ_RE"],
}

method_tag = "method"
trial_tag = "trial"
# create a merged tag
merged_tag = "instance"
auc_tag = "-auc"

variant_tag_names = [
    "RL",
    "Encoder",
    "Len",
    "Inputs",
    "Arch",
]

diagnosis_tags = [
    ## loss/gradient tags
    # ("rl_loss/qf1_loss", "q_loss"),
    # ("rl_loss/policy_loss", "pi_loss"), # pi loss can be negative, so we don't plot it
    # for shared ac
    ("rl_loss/rnn_grad_norm", "rnn_grad_norm"),
    # for separate ac
    ("rl_loss/q_rnn_grad_norm", "q_rnn_grad_norm"),
    ("rl_loss/pi_rnn_grad_norm", "pi_rnn_grad_norm"),
]

ours_name = "Ours: recurrent model-free RL"
baseline_names = ["sac_gru", "sac_lstm", "sacd_lstm", "td3_gru", "td3_lstm"]


def get_variant_tags(trial_str, max_episode_len):
    v = dict()
    # find RL tag
    if "sac_" in trial_str:
        v["RL"] = "sac"
    elif "sacd_" in trial_str:
        v["RL"] = "sacd"
    elif "td3" in trial_str:
        v["RL"] = "td3"
    if "lstm" in trial_str:
        v["Encoder"] = "lstm"
    elif "gru" in trial_str:
        v["Encoder"] = "gru"
    if "shared" in trial_str:
        v["Arch"] = "shared"
    else:
        v["Arch"] = "separate"
    v["Len"] = int(trial_str[trial_str.index("len-") + 4 :].split("/")[0])
    if v["Len"] == -1:
        v["Len"] = max_episode_len
    v["Inputs"] = trial_str.split("/")[-3]

    return v


def get_env_tag(path):
    if "pomdp" in path:
        if "/P/" in path:
            suffix = "P"
        else:
            suffix = "V"
        if "Ant" in path:
            prefix = "Ant"
        elif "HalfCheetah" in path:
            prefix = "Cheetah"
        elif "Hopper" in path:
            prefix = "Hopper"
        elif "Walker" in path:
            prefix = "Walker"
        else:
            raise ValueError
        return prefix + "-" + suffix
    elif "meta" in path:
        if "PointRobotSparse" in path:
            return "Semi-Circle"
        elif "Wind" in path:
            return "Wind"
        elif "HalfCheetahVel" in path:
            return "Cheetah-Vel"
        elif "AntDir" in path:
            return "Ant-Dir"
        elif "CheetahDir" in path:
            return "Cheetah-Dir"
        elif "HumanoidDir" in path:
            return "Humanoid-Dir"
    elif "rmdp" in path:
        if "HalfCheetah" in path:
            prefix = "Cheetah"
        elif "Hopper" in path:
            prefix = "Hopper"
        elif "Walker" in path:
            prefix = "Walker"
        return prefix + "-" + "Robust"
    elif "generalize" in path:
        if "Cheetah" in path:
            prefix = "Cheetah"
        elif "Hopper" in path:
            prefix = "Hopper"
        return prefix + "-" + "Generalize"


# the content will not be used, not important
specialized_tags = {
    "onpolicy-varibad": {
        "RL": "ppo",
        "Encoder": "gru",
        "Len": "-1",
        "Inputs": "oard",
        "Arch": "separate",
    },
    "oracle_ppo": {
        "RL": "ppo",
        "Encoder": "mlp",
        "Len": "1",
        "Inputs": "s",
        "Arch": "separate",
    },  # from onpolicy varibad data
    "offpolicy-varibad": {
        "RL": "sac",
        "Encoder": "gru",
        "Len": "-1",
        "Inputs": "oard",
        "Arch": "separate",
    },
    "re-varibad": {
        "RL": "sac",
        "Encoder": "gru",
        "Len": "-1",
        "Inputs": "oard",
        "Arch": "separate",
    },  # discarded as re-implemented off-varibad
    "rl2": {
        "RL": "ppo",
        "Encoder": "gru",
        "Len": "-1",
        "Inputs": "oard",
        "Arch": "separate",
    },
    "VRM": {
        "RL": "sac",
        "Encoder": "lstm",
        "Len": "64",
        "Inputs": "oar",
        "Arch": "separate",
    },
    "MRPO": {
        "RL": "ppo",
        "Encoder": "no",
        "Len": "1",
        "Inputs": "s",
        "Arch": "separate",
    },
    "ppo_gru": {
        "RL": "ppo",
        "Encoder": "gru",
        "Len": "128",
        "Inputs": "o",
        "Arch": "shared",
    },
    "a2c_gru": {
        "RL": "a2c",
        "Encoder": "gru",
        "Len": "5",
        "Inputs": "o",
        "Arch": "shared",
    },
    "sac_mlp": {
        "RL": "sac",
        "Encoder": "no",
        "Len": "1",
        "Inputs": "o",
        "Arch": "separate",
    },
    "td3_mlp": {
        "RL": "td3",
        "Encoder": "no",
        "Len": "1",
        "Inputs": "o",
        "Arch": "separate",
    },
    "Markovian_sac": {
        "RL": "sac",
        "Encoder": "no",
        "Len": "1",
        "Inputs": "o",
        "Arch": "separate",
    },
    "Markovian_td3": {
        "RL": "td3",
        "Encoder": "no",
        "Len": "1",
        "Inputs": "o",
        "Arch": "separate",
    },
    "oracle_sac": {
        "RL": "sac",
        "Encoder": "no",
        "Len": "1",
        "Inputs": "s",
        "Arch": "separate",
    },
    "oracle_td3": {
        "RL": "td3",
        "Encoder": "no",
        "Len": "1",
        "Inputs": "s",
        "Arch": "separate",
    },
    "oracle_rnn": {
        "RL": "RL",
        "Encoder": "rnn",
        "Len": "64",
        "Inputs": "s",
        "Arch": "separate",
    },
    "td3_2lstm": {
        "RL": "td3",
        "Encoder": "2lstm",
        "Len": "64",
        "Inputs": "o",
        "Arch": "separate",
    },
    "td3_2gru": {
        "RL": "td3",
        "Encoder": "2gru",
        "Len": "64",
        "Inputs": "oa",
        "Arch": "separate",
    },
    "td3_pfgru": {
        "RL": "td3",
        "Encoder": "pfgru",
        "Len": "64",
        "Inputs": "oa",
        "Arch": "separate",
    },
}

# best plots
no_dash = (1, 0)
dashes1 = (2, 1)
dashes2 = (4, 1)

curve_style_dict = {
    # key: (color, dash)
    ours_name: ("blue", no_dash),
    "oracle_rnn": ("magenta", no_dash),
    "VRM": ("green", no_dash),
    "ppo_gru": ("magenta", no_dash),
    "a2c_gru": ("yellow", no_dash),
    "MRPO": ("red", no_dash),
    "onpolicy-varibad": ("green", no_dash),
    "offpolicy-varibad": ("red", no_dash),
    "re-varibad": ("yellow", no_dash),
    "rl2": ("magenta", no_dash),
    "oracle_ppo": ("yellow", no_dash),
    "A2C-RC": ("yellow", no_dash),
    "EPOpt-PPO-FF": ("red", no_dash),
    "Oracle": ("orange", no_dash),
    "Markovian": ("black", no_dash),
    "Random": ("purple", no_dash),
    "sac_mlp": ("black", no_dash),
    "td3_mlp": ("purple", no_dash),
    "Markovian_sac": ("black", no_dash),
    "Markovian_td3": ("purple", no_dash),
    "oracle_sac": ("orange", no_dash),
    "oracle_td3": ("pink", no_dash),
    "td3_2lstm": ("brown", no_dash),
    "td3_2gru": ("brown", no_dash),
    "td3_pfgru": ("magenta", no_dash),
    "IMPALA+SR": ("red", no_dash),
}

# ablation plots
variant_colors = {
    "RL": {"sac": "red", "sacd": "red", "td3": "blue"},
    "Encoder": {"gru": "red", "lstm": "blue"},
    "Inputs": {"o": "yellow", "or": "green", "oa": "red", "oar": "blue"},
    "Arch": {"shared": "red", "separate": "blue"},
}

table_results = {
    "Ant-Dir": {
        "return": {"Random": -48.247},
    },
    "Cheetah-Dir": {
        "return": {"Random": -39.3},
    },
    "Humanoid-Dir": {
        "return": {"Random": 296.991},
    },
    "Cheetah-Vel": {
        "return": {"Oracle": -79.692, "Markovian": -351.654, "Random": -646.3391},
    },
    "Semi-Circle": {
        "return": {"Oracle": 103.347, "Markovian": 12.433, "Random": 3.0},
    },
    "Wind": {
        "return": {"Oracle": 63.695, "Markovian": 18.351, "Random": 0.0},
    },
    "Ant-P": {
        "return": {"Oracle": 3394, "Markovian": 1132.946, "Random": 372.6779},
    },
    "Ant-V": {
        "return": {"Oracle": 3394, "Markovian": 645.05, "Random": 372.6779},
    },
    "Cheetah-P": {
        "return": {"Oracle": 2994, "Markovian": 870.48, "Random": -1256.6604},
    },
    "Cheetah-V": {
        "return": {"Oracle": 2994, "Markovian": 517.946, "Random": -1256.6604},
    },
    "Hopper-P": {
        "return": {"Oracle": 2434, "Markovian": 477.911, "Random": 19.9634},
    },
    "Hopper-V": {
        "return": {"Oracle": 2434, "Markovian": 228.366, "Random": 19.9634},
    },
    "Walker-P": {
        "return": {"Oracle": 2225, "Markovian": 471.815, "Random": 16.5319},
    },
    "Walker-V": {
        "return": {"Oracle": 2225, "Markovian": 214.45, "Random": 16.5319},
    },
    "Cheetah-Robust": {
        "return_avg": {"Random": 1.3098},  # "Oracle": 1372.335, "Markovian": 1363.131,
        "return_worst": {"Random": -5.6627},  # "Oracle": 896.129, "Markovian": 807.229,
    },
    "Hopper-Robust": {
        "return_avg": {"Random": 21.262},  # "Oracle": 1409.703, "Markovian": 1438.069,
        "return_worst": {"Random": 16.703},  # "Oracle": 427.483, "Markovian": 476.617,
    },
    "Walker-Robust": {
        "return_avg": {"Random": 16.3323},  # "Oracle": 1146.887, "Markovian": 1159.432,
        "return_worst": {"Random": 12.3943},  # "Oracle": 545.257, "Markovian": 570.559,
    },
    "Cheetah-Generalize": {
        "Default": {"A2C-RC": 0.8806, "EPOpt-PPO-FF": 0.9976},
        "Interpolation": {
            "A2C-RC": 0.7470,
            "EPOpt-PPO-FF": 0.9928,
            # "Oracle": 0.255, "Markovian": 0.642,
            "Random": 0.0,
        },
        "Extrapolation": {
            "A2C-RC": 0.4296,
            "EPOpt-PPO-FF": 0.5341,
            # "Oracle": 0.1323, "Markovian": 0.293,
            "Random": 0.0,
        },
    },
    "Hopper-Generalize": {
        "Default": {"A2C-RC": 0.1534, "EPOpt-PPO-FF": 0.5862},
        "Interpolation": {
            "A2C-RC": 0.1038,
            "EPOpt-PPO-FF": 0.8078,
            # "Oracle": 0.466, "Markovian": 0.307,
            "Random": 0.0,
        },
        "Extrapolation": {
            "A2C-RC": 0.0131,
            "EPOpt-PPO-FF": 0.2139,
            # "Oracle": 0.2847, "Markovian": 0.245,
            "Random": 0.0,
        },
    },
    "Delayed-Catch": {
        "return": {"IMPALA+SR": 3.1, "Random": 5.93},  # at 2.5M
    },
    "Key-to-Door": {
        "success_rate": {"IMPALA+SR": 0.57, "Random": 0.022},  # at 4M
    },
}
