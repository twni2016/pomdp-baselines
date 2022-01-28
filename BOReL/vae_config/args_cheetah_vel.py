import argparse
import torch
from utils.cli import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="HalfCheetahVel-v0")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--max-rollouts-per-task", default=2)
    parser.add_argument("--num-trajs-per-task", type=int, default=None)
    parser.add_argument("--hindsight-relabelling", type=int, default=True)
    # parser.add_argument('--hindsight-relabelling', type=int, default=False)

    parser.add_argument("--num-iters", default=100)
    parser.add_argument("--tasks-batch-size", default=4)
    parser.add_argument("--vae-batch-num-rollouts-per-task", default=8)
    parser.add_argument(
        "--vae-lr",
        type=float,
        default=0.0003,
        help="learning rate for VAE (default: 3e-4)",
    )
    parser.add_argument(
        "--kl-weight", type=float, default=0.05, help="weight for the KL term"
    )
    parser.add_argument(
        "--vae-batch-num-elbo-terms",
        default=None,
        help="for how many timesteps to compute the ELBO; None uses all",
    )
    # - encoder
    parser.add_argument(
        "--encoder_type", type=str, default="rnn", help="choose: rnn, tcn, deepset"
    )
    parser.add_argument(
        "--task-embedding-size",
        type=int,
        default=5,
        help="dimensionality of latent space",
    )
    parser.add_argument("--aggregator-hidden-size", type=int, default=128)
    parser.add_argument("--layers-before-aggregator", nargs="+", type=int, default=[])
    parser.add_argument("--layers-after-aggregator", nargs="+", type=int, default=[])
    parser.add_argument("--action-embedding-size", type=int, default=16)
    parser.add_argument("--state-embedding-size", type=int, default=32)
    parser.add_argument("--reward-embedding-size", type=int, default=16)

    # - decoder: rewards
    parser.add_argument("--decode-reward", default=True, help="use reward decoder")
    parser.add_argument(
        "--input-prev-state", default=True, help="use prev state for rew pred"
    )
    parser.add_argument(
        "--input-action", default=True, help="use prev action for rew pred"
    )
    parser.add_argument(
        "--reward-decoder-layers", nargs="+", type=int, default=[32, 32]
    )
    parser.add_argument(
        "--rew-pred-type",
        type=str,
        default="deterministic",
        help="choose from: bernoulli, deterministic",
    )
    parser.add_argument(
        "--multihead-for-reward",
        default=False,
        help="one head per reward pred (i.e. per state)",
    )
    parser.add_argument("--rew-loss-coeff", type=float, default=1.0)

    # - decoder: state transitions
    parser.add_argument("--decode-state", default=False)
    parser.add_argument("--state-loss-coeff", type=float, default=1.0)

    # - decoder: ground-truth task (after Humplik et al. 2019)
    parser.add_argument("--decode-task", default=False)
    parser.add_argument("--task-loss-coeff", default=1.0)

    # --- ABLATIONS ---
    parser.add_argument("--disable-decoder", default=False)
    parser.add_argument("--disable-stochasticity-in-latent", default=False)
    parser.add_argument("--kl-to-gauss-prior", default=False)
    parser.add_argument("--learn-prior", default=False)
    parser.add_argument(
        "--decode-only-past",
        default=False,
        help="whether to decode future observations",
    )

    parser.add_argument("--log-interval", default=1)
    parser.add_argument("--save-interval", default=5)
    parser.add_argument("--eval-interval", default=20)

    parser.add_argument("--main-data-dir", default="./batch_data")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--save-dir-prefix", default="relabel")
    # parser.add_argument('--save-dir-prefix', default='no_relabel')
    parser.add_argument("--log-tensorboard", default=True)
    parser.add_argument("--save-model", default=True)
    parser.add_argument("--save-dir", default="./trained_vae")
    parser.add_argument("--use-gpu", default=True)

    args = parser.parse_args(rest_args)

    return args
