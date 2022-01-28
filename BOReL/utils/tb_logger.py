import json
import datetime
import os
from tensorboardX import SummaryWriter
from torchkit import pytorch_utils as ptu


class TBLogger:
    def __init__(self, args):

        # initialise name of the file (optional(prefix) + seed + start time)
        cql_ext = "_cql" if "use_cql" in args and args.use_cql else ""
        if hasattr(args, "output_file_prefix"):
            self.output_name = (
                args.output_file_prefix
                + cql_ext
                + "__"
                + str(args.seed)
                + "__"
                + datetime.datetime.now().strftime("%d_%m_%H_%M_%S")
            )
        else:
            self.output_name = (
                str(args.seed)
                + "__"
                + datetime.datetime.now().strftime("%d_%m_%H_%M_%S")
            )

        # get path to log directory (and create it if necessary)
        try:
            log_dir = args.results_log_dir
        except AttributeError:
            log_dir = args["results_log_dir"]

        if log_dir is None:
            log_dir = os.path.abspath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
            )
            log_dir = os.path.join(log_dir, "logs")

        if not os.path.exists(log_dir):
            try:
                os.mkdir(log_dir)
            except:
                dir_path_head, dir_path_tail = os.path.split(log_dir)
                if len(dir_path_tail) == 0:
                    dir_path_head, dir_path_tail = os.path.split(dir_path_head)
                os.mkdir(dir_path_head)
                os.mkdir(log_dir)

        # create a subdirectory for the environment
        try:
            env_dir = os.path.join(log_dir, "{}".format(args.env_name))
        except:
            env_dir = os.path.join(log_dir, "{}".format(args["env_name"]))
        if not os.path.exists(env_dir):
            os.makedirs(env_dir)

        # create a subdirectory for the exp_label (usually the method name)
        # exp_dir = os.path.join(env_dir, exp_label)
        # if not os.path.exists(exp_dir):
        #     os.makedirs(exp_dir)

        # finally, get full path of where results are stored
        self.full_output_folder = os.path.join(env_dir, self.output_name)

        self.writer = SummaryWriter(self.full_output_folder)

        print("logging under", self.full_output_folder)

        with open(
            os.path.join(self.full_output_folder, "online_config.json"), "w"
        ) as f:
            try:
                config = {k: v for (k, v) in vars(args).items() if k != "device"}
            except:
                config = args
            config.update(device=ptu.device.type)
            json.dump(config, f, indent=2)
