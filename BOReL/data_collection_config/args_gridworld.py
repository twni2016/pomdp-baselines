import argparse
import torch
from utils.cli import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="GridNavi-v2")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--policy", default="dqn")
    parser.add_argument("--max-rollouts-per-task", default=4)
    parser.add_argument("--policy-buffer-size", default=1e6)
    parser.add_argument("--dqn-layers", default=[16, 16])
    parser.add_argument("--policy-lr", default=3e-4)
    parser.add_argument("--dqn-epsilon-init", default=1.0)
    # parser.add_argument('--dqn-epsilon-final', default=0.1)
    parser.add_argument("--dqn-epsilon-final", default=0.0)
    # parser.add_argument('--dqn-exploration-iters', default=100)
    parser.add_argument("--dqn-exploration-iters", default=50)
    parser.add_argument("--gamma", default=0.99)
    parser.add_argument("--soft-target-tau", default=0.005)
    # parser.add_argument('--num-iters', default=200)
    parser.add_argument("--num-iters", default=100)
    # parser.add_argument('--num-init-rollouts-pool', default=30)
    parser.add_argument("--num-init-rollouts-pool", default=10)
    parser.add_argument("--num-rollouts-per-iter", default=5)
    parser.add_argument("--batch-size", default=256)
    # parser.add_argument('--rl-updates-per-iter', default=500)
    parser.add_argument("--rl-updates-per-iter", default=200)
    parser.add_argument("--eval-deterministic", default=True)

    # Ablation - initial state distribution
    parser.add_argument("--modify-init-state-dist", default=True)

    parser.add_argument("--log-interval", default=10)
    parser.add_argument("--save-interval", default=20)

    parser.add_argument("--save-buffer", default=True)  # TODO - if false, crashes
    parser.add_argument("--save-models", default=False)
    parser.add_argument("--main-save-dir", default="./batch_data")
    parser.add_argument("--log-tensorboard", default=False)
    parser.add_argument("--save-dir", default="data")
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

    args = parser.parse_args(rest_args)

    return args
