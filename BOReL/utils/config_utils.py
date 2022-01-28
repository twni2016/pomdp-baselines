import os
import json
import torchkit.pytorch_utils as ptu
from argparse import Namespace


def save_config_file(args, path):
    with open(os.path.join(path, "online_config.json"), "w") as f:
        try:
            config = {k: v for (k, v) in vars(args).items() if k != "device"}
        except:
            config = args
        config.update(device=ptu.device.type)
        json.dump(config, f, indent=2)


def load_config_file(path):
    with open(os.path.join(path)) as f:
        config = json.load(f)
        args = Namespace(**config)
    return args


def merge_configs(*args):
    """
    Merges multiple Namespace objects created via parser.parse_args()
    """
    merged_arguments = args[0]
    for arg in args:
        merged_arguments.__dict__.update(arg.__dict__)

    return merged_arguments
