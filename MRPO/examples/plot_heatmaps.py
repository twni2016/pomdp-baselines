"""Plot heatmaps of evaluation results."""
import argparse
import itertools
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import scipy.spatial

CELL_AGGREGATIONS = {
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("results", type=str, help="Experiments results file")
    parser.add_argument("--grid-size", type=int, default=10, help="Heatmap grid size")
    parser.add_argument(
        "--cell-aggregation", type=str, choices=CELL_AGGREGATIONS.keys(), default="mean"
    )
    parser.add_argument("--scale-min", type=int)
    parser.add_argument("--scale-max", type=int)
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cell_aggregation = CELL_AGGREGATIONS[args.cell_aggregation]

    with open(args.results, "r") as results_file:
        # Read results for each experiment (in its own line).
        for index, experiment in enumerate(results_file):
            try:
                experiment = json.loads(experiment)
            except ValueError:
                print("WARNING: Skipping malformed experiment result.")
                continue

            print(
                "Processing result {index}/{trained_on}/{evaluated_on}...".format(
                    index=index, **experiment
                )
            )

            # Assume all episodes vary the same parameters.
            episodes = experiment["episodes"]
            parameters = set(episodes[0]["environment"].keys())
            parameters.remove("id")
            parameters.remove("world")

            # Remove non-numeric parameters.
            for parameter in parameters.copy():
                values = np.asarray(
                    [episode["environment"][parameter] for episode in episodes]
                )
                if not np.issubdtype(values.dtype, np.number):
                    parameters.remove(parameter)

            # Sort parameters alphabetically for a consistent order.
            parameters = sorted(parameters)

            # Rewards.
            rewards = np.asarray([episode["reward"] for episode in episodes])

            # Colormap.
            colormap = plt.cm.rainbow
            colornorm = matplotlib.colors.Normalize(
                vmin=args.scale_min or np.min(rewards),
                vmax=args.scale_max or np.max(rewards),
            )

            # Compute all-pairs heatmaps.
            items = len(parameters)
            figure, axes = plt.subplots(
                items, items, sharex="col", sharey="row", figsize=(12, 12)
            )
            if items == 1:
                axes = np.asarray([axes]).reshape([1, 1])

            for row, param_a in enumerate(parameters):
                axes[0, row].set_title(param_a)

                for col, param_b in enumerate(parameters):
                    axes[col, 0].set_ylabel(param_b, size="large")

                    values_a = np.asarray(
                        [float(episode["environment"][param_a]) for episode in episodes]
                    )
                    values_b = np.asarray(
                        [float(episode["environment"][param_b]) for episode in episodes]
                    )

                    # Sort by rewards.
                    rewards_idx = sorted(
                        np.arange(rewards.shape[0]), key=lambda index: rewards[index]
                    )
                    rewards = rewards[rewards_idx]
                    values_a = values_a[rewards_idx]
                    values_b = values_b[rewards_idx]

                    zmin = rewards.min()
                    zmax = rewards.max()

                    ax = axes[col, row]

                    # Plot heatmap.
                    heatmap = ax.hexbin(
                        values_a,
                        values_b,
                        rewards,
                        cmap=colormap,
                        norm=colornorm,
                        gridsize=args.grid_size,
                        reduce_C_function=cell_aggregation,
                    )

            # Plot colorbar.
            figure.colorbar(heatmap, ax=axes.ravel().tolist())

            plt.suptitle(
                "Model: $\\bf{{{model[name]}}}$ Trained: {trained_on}\n"
                "Evaluated: {evaluated_on}\n"
                "Episodes: {n_episodes} Mean: {mean:.2f} Median: {median:.2f} Min: {min:.2f} Max: {max:.2f}\n"
                "Grid size: {grid_size}x{grid_size} Cell aggregation: {cell_aggregation}"
                "".format(
                    n_episodes=len(episodes),
                    mean=np.mean(rewards),
                    median=np.median(rewards),
                    min=np.min(rewards),
                    max=np.max(rewards),
                    grid_size=args.grid_size,
                    cell_aggregation=args.cell_aggregation,
                    **experiment
                )
            )
            plt.savefig(
                os.path.join(
                    args.output,
                    "heatmap-train-{trained_on}-test-{evaluated_on}-{index:02d}.png".format(
                        index=index, **experiment
                    ),
                )
            )
            plt.close()
