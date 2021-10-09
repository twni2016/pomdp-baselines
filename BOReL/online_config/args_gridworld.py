import argparse
import torch
from utils.cli import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # --- GENERAL ---
    parser.add_argument(
        "--output-file-prefix", default="dqn", help="prefix to output file name"
    )
    parser.add_argument(
        "--load-model",
        type=boolean_argument,
        default=False,
        help="whether to load trained model",
    )

    parser.add_argument(
        "--policy-buffer-size", type=int, default=1e6, help="buffer size for RL updates"
    )
    parser.add_argument(
        "--vae-buffer-size", type=int, default=1e5, help="buffer size for VAE updates"
    )
    # --- ENV ---
    parser.add_argument(
        "--env-name", default="GridNavi-v2", help="environment to train on"
    )
    parser.add_argument("--max-rollouts-per-task", type=int, default=4)
    parser.add_argument(
        "--num-tasks", type=int, default=21, help="number of goals in environment"
    )
    parser.add_argument(
        "--num-train-tasks", type=int, default=16, help="number of tasks for train"
    )
    parser.add_argument(
        "--num-eval-tasks", type=int, default=5, help="number of tasks for evaluation"
    )

    # --- TRAINING ---
    parser.add_argument(
        "--fixed-latent-params",
        type=boolean_argument,
        default=False,
        help="whether to set fixed latent parameters to check capacity of RL",
    )  # DEBUG only

    parser.add_argument(
        "--meta-batch",
        type=int,
        default=16,
        help="number of tasks to average the gradient across",
    )
    parser.add_argument(
        "--num-tasks-sample",
        type=int,
        default=5,
        help="number of tasks to collect rollouts per iter",
    )

    parser.add_argument(
        "--num-iters", type=int, default=30000, help="number meta-training iterates"
    )
    parser.add_argument(
        "--rl-updates-per-iter",
        type=int,
        default=250,
        help="number of RL steps per iteration",
    )
    parser.add_argument(
        "--vae-updates-per-iter",
        type=int,
        default=20,
        help="number of VAE steps per iteration",
    )
    parser.add_argument(
        "--num-rollouts-per-iter",
        type=int,
        default=5,
        help="number of rollouts to collect per task",
    )
    parser.add_argument(
        "--num-init-rollouts-pool",
        type=int,
        default=20,
        help="number of initial rollouts collect per task, before training begins",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="number of transitions in RL batch (per task)",
    )
    parser.add_argument(
        "--vae-batch-num-rollouts-per-task",
        type=int,
        default=2,
        help="number of rollouts in VAE batch (per task)",
    )
    parser.add_argument(
        "--vae-batch-num-elbo-terms",
        type=int,
        default=None,
        help="for how many timesteps to compute the ELBO; None uses all",
    )

    # --- POLICY ---

    # network
    parser.add_argument("--dqn-layers", nargs="+", default=[100, 100])

    # algo
    parser.add_argument("--policy", type=str, default="dqn", help="choose: dqn, ddqn")

    # dqn specific
    parser.add_argument(
        "--dqn-alpha",
        type=float,
        default=0.99,
        help="RMSprop optimizer alpha (default: 0.99)",
    )
    parser.add_argument(
        "--dqn-eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument(
        "--dqn-epsilon-init",
        type=float,
        default=1.0,
        help="initial explor. param (default: 1.0)",
    )
    parser.add_argument(
        "--dqn-epsilon-final",
        type=float,
        default=0.1,
        help="final explor. param (default: 0.1)",
    )
    parser.add_argument(
        "--dqn-exploration-iters",
        type=float,
        default=1000,
        help="iters from episolon_init to epsilon_final (default: 1000)",
    )
    parser.add_argument(
        "--soft-target-tau",
        type=float,
        default=0.005,
        help="soft target network update (default: 5e-3)",
    )

    # other hyperparameters
    parser.add_argument(
        "--policy-lr",
        type=float,
        default=0.00007,
        help="learning rate for RL (default: 7e-5)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )

    parser.add_argument(
        "--use-proper-time-limits", type=boolean_argument, default=False
    )

    parser.add_argument(
        "--oracle-belief-rewards",
        type=boolean_argument,
        default=True,
        help="whether to use oracle R+ in policy update",
    )
    parser.add_argument(
        "--switch-to-belief-reward",
        type=int,
        default=0,
        help="when to switch from R to R+; None is to not switch",
    )
    parser.add_argument(
        "--num-belief-samples",
        type=int,
        default=40,
        help="number of latent samples to estimate R+; if oracle, not relevant",
    )

    # --- VAE TRAINING ---

    # general
    parser.add_argument(
        "--pretrain-len", type=boolean_argument, default=0, help="num. of vae updates"
    )
    parser.add_argument(
        "--vae-lr",
        type=float,
        default=0.0003,
        help="learning rate for VAE (default: 3e-4)",
    )
    parser.add_argument(
        "--kl_weight", type=float, default=1.0, help="weight for the KL term"
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
    parser.add_argument(
        "--aggregator-hidden-size",
        type=int,
        default=64,
        help="dimensionality of hidden state of the rnn",
    )
    parser.add_argument("--layers-before-aggregator", nargs="+", type=int, default=[])
    parser.add_argument("--layers-after-aggregator", nargs="+", type=int, default=[])
    parser.add_argument("--action-embedding-size", type=int, default=0)
    parser.add_argument("--state-embedding-size", type=int, default=32)
    parser.add_argument("--reward-embedding-size", type=int, default=8)

    # - decoder: rewards
    parser.add_argument(
        "--decode-reward",
        type=boolean_argument,
        default=True,
        help="use reward decoder",
    )
    parser.add_argument(
        "--input-prev-state",
        type=boolean_argument,
        default=False,
        help="use prev state for rew pred",
    )
    parser.add_argument(
        "--input-action",
        type=boolean_argument,
        default=False,
        help="use prev action for rew pred",
    )
    parser.add_argument(
        "--reward-decoder-layers", nargs="+", type=int, default=[32, 32]
    )
    parser.add_argument(
        "--rew-pred-type",
        type=str,
        default="bernoulli",
        help="choose from: bernoulli, gaussian, deterministic",
    )
    parser.add_argument(
        "--rew-loss-fn", type=str, default="BCE", help="choose from: BCE, FL"
    )
    parser.add_argument(
        "--multihead-for-reward",
        type=boolean_argument,
        default=True,
        help="one head per reward pred (i.e. per state)",
    )
    parser.add_argument(
        "--rew-loss-coeff", type=float, default=1.0, help="weight for reward loss"
    )

    # - decoder: state transitions
    parser.add_argument(
        "--decode-state", type=boolean_argument, default=False, help="use state decoder"
    )
    parser.add_argument(
        "--state-loss-coeff",
        type=float,
        default=1.0,
        help="weight for state loss (vs reward loss)",
    )

    # - decoder: ground-truth task ("varibad oracle", after Humplik et al. 2019)
    parser.add_argument(
        "--decode-task", type=boolean_argument, default=False, help="use state decoder"
    )
    parser.add_argument(
        "--task-loss-coeff",
        type=float,
        default=1.0,
        help="weight for task decoding loss (vs reward loss)",
    )

    # --- ABLATIONS ---

    parser.add_argument("--disable-decoder", type=boolean_argument, default=False)
    parser.add_argument(
        "--disable-stochasticity-in-latent", type=boolean_argument, default=False
    )
    parser.add_argument(
        "--sample-embeddings",
        type=boolean_argument,
        default=False,
        help="sample the embedding (otherwise: pass mean)",
    )
    parser.add_argument(
        "--vae-loss-coeff",
        type=float,
        default=1.0,
        help="weight for VAE loss (vs RL loss)",
    )
    parser.add_argument("--kl-to-gauss-prior", type=boolean_argument, default=False)
    parser.add_argument("--learn-prior", type=boolean_argument, default=False)
    parser.add_argument(
        "--decode-only-past",
        type=boolean_argument,
        default=False,
        help="whether to decode future observations",
    )
    parser.add_argument(
        "--condition-policy-on-state",
        type=boolean_argument,
        default=True,
        help="after the encoder, add the env state to the latent space",
    )

    # --- OTHERS ---

    # logging, saving, evaluation
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="log interval, one log per n iterations (default: 20)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=200,
        help="save models interval, every # iterations (default: 50)",
    )
    parser.add_argument(
        "--agent-log-dir",
        default="tmp/gym/",
        help="directory to save agent logs (default: /tmp/gym)",
    )
    parser.add_argument(
        "--results-log-dir",
        default=None,
        help="directory to save agent logs (default: ./data)",
    )

    parser.add_argument(
        "--log-tensorboard",
        type=boolean_argument,
        default=True,
        help="whether to use tb logger",
    )

    # general settings
    parser.add_argument(
        "--seed", type=int, default=73, help="random seed (default: 73)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8097,
        help="port to run the server on (default: 8097)",
    )

    # gpu settings
    parser.add_argument(
        "--use-gpu", type=boolean_argument, default=True, help="whether to use gpu"
    )
    args = parser.parse_args(rest_args)

    args.cuda = torch.cuda.is_available()

    return args
