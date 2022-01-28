import os
import argparse
import torch
from torchkit.pytorch_utils import set_gpu_mode
from models.vae import VAE
from offline_metalearner import OfflineMetaLearner
import utils.config_utils as config_utl
from utils import offline_utils as off_utl
from offline_config import (
    args_ant_semicircle_sparse,
    args_cheetah_vel,
    args_point_robot_sparse,
    args_gridworld,
)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env-type', default='gridworld')
    # parser.add_argument('--env-type', default='point_robot_sparse')
    # parser.add_argument('--env-type', default='cheetah_vel')
    parser.add_argument("--env-type", default="ant_semicircle_sparse")
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    # --- GridWorld ---
    if env == "gridworld":
        args = args_gridworld.get_args(rest_args)
    # --- PointRobot ---
    elif env == "point_robot_sparse":
        args = args_point_robot_sparse.get_args(rest_args)
    # --- Mujoco ---
    elif env == "cheetah_vel":
        args = args_cheetah_vel.get_args(rest_args)
    elif env == "ant_semicircle_sparse":
        args = args_ant_semicircle_sparse.get_args(rest_args)

    set_gpu_mode(torch.cuda.is_available() and args.use_gpu)

    vae_args = config_utl.load_config_file(
        os.path.join(
            args.vae_dir, args.env_name, args.vae_model_name, "online_config.json"
        )
    )
    args = config_utl.merge_configs(
        vae_args, args
    )  # order of input to this function is important

    # Transform data BAMDP (state relabelling)
    if args.transform_data_bamdp:
        # load VAE for state relabelling
        vae_models_path = os.path.join(
            args.vae_dir, args.env_name, args.vae_model_name, "models"
        )
        vae = VAE(args)
        off_utl.load_trained_vae(vae, vae_models_path)
        # load data and relabel
        save_data_path = os.path.join(
            args.main_data_dir, args.env_name, args.relabelled_data_dir
        )
        os.makedirs(save_data_path)
        dataset, goals = off_utl.load_dataset(
            data_dir=args.data_dir, args=args, arr_type="numpy"
        )
        bamdp_dataset = off_utl.transform_mdps_ds_to_bamdp_ds(dataset, vae, args)
        # save relabelled data
        off_utl.save_dataset(save_data_path, bamdp_dataset, goals)

    learner = OfflineMetaLearner(args)

    learner.train()


if __name__ == "__main__":
    main()
