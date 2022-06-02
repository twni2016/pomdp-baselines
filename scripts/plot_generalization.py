import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import pandas as pd
import numpy as np
import os, sys
from scripts.constants import *
from absl import flags

FLAGS = flags.FLAGS

## Hparams
flags.DEFINE_string("merged_path", None, "merged_path")
flags.DEFINE_list("csv_paths", None, "csv paths")
flags.DEFINE_integer("max_x", None, "max value of x")
flags.DEFINE_integer("window_size", 20, "window size of the plot")
flags.DEFINE_float("last_steps_ratio", 0.80, "for measureing top k")
flags.DEFINE_list("factors", None, "selected single factors")
flags.DEFINE_string("best_variant", None, "best variant of our method if given")
flags.DEFINE_list("other_methods", None, "selective other methods to show if given")
flags.DEFINE_string("name", None, "for plot tabular results")
flags.FLAGS(sys.argv)
print(FLAGS.flags_into_string())


### 1. Preprocess

assert len(FLAGS.csv_paths) == 2
df_D = pd.read_csv(FLAGS.csv_paths[0])
df_R = pd.read_csv(FLAGS.csv_paths[1])
assert len(df_D) == len(df_R)

variant_tag_names = [
    variant_tag_name
    for variant_tag_name in variant_tag_names
    if variant_tag_name in df_D.columns
]
if FLAGS.factors is None:
    FLAGS.factors = variant_tag_names
assert set(FLAGS.factors) <= set(variant_tag_names)  # subset check

# join (merge) two dfs
df_D[merged_tag] = df_D[variant_tag_names].astype(str).agg("-".join, axis=1)
df_R[merged_tag] = df_R[variant_tag_names].astype(str).agg("-".join, axis=1)
df_D = df_D.reset_index()  # add column 'index' to keep env steps order for sorting
df_R = df_R.reset_index()
df_D = df_D.sort_values(by=["instance", "index"], ignore_index=True)
df_R = df_R.sort_values(by=["instance", "index"], ignore_index=True)
# HACK: use concat to join the subtable... make one seed trial has both D and R results...
df = pd.concat([df_D, df_R[["succ_RR", "succ_RE"]]], axis=1)

# create new tags
for (new_tag, raw_tags) in generalization_tags.items():
    df[new_tag] = df[raw_tags].mean(axis=1)  # avg over raw tags

os.makedirs(FLAGS.merged_path, exist_ok=True)

# extract the tags
x_tag = x_tag[1]
key_of_interests = list(generalization_tags.keys())

if FLAGS.max_x is not None:
    df = df.loc[df[x_tag] <= FLAGS.max_x]  # set max_x

# smoothing
for key in key_of_interests:
    df[key] = df.groupby([*variant_tag_names, trial_tag])[key].transform(
        lambda x: x.rolling(FLAGS.window_size, min_periods=1).mean()  # rolling mean
    )


### 2. plot single factor

# make a square-like plot, show Interpolation and Extrapolation only, so use [1:]
num_plots = len(key_of_interests[1:]) * max(1, len(FLAGS.factors))
cols = int(np.ceil(np.sqrt(num_plots)))
rows = int(np.ceil(num_plots / cols))

# seaborn plot
sns.set(font_scale=2.0)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4))
axes = (
    axes.flatten() if isinstance(axes, np.ndarray) else [axes]
)  # make it as a flat list

# use lineplot that has average curve (for same x-value) with 95% confidence interval on y-value
# https://seaborn.pydata.org/generated/seaborn.lineplot.html
# has at most 3 independent dims to plot, using hue and style. But recommend to use at most 2 dims,
# 	by setting hue and style the same key
# NOTE: any seaborn function has argument ax to support subplots

df_ours = df.loc[df["method"] == "ours"]
df_others = df.loc[df["method"] != "ours"]
ax_id = 0
for key in key_of_interests[1:]:
    for variant_tag_name in FLAGS.factors:

        sns.lineplot(
            ax=axes[ax_id],
            data=df_ours,
            x=x_tag,
            y=key,
            hue=variant_tag_name,
            # hue_order=order,
            # style=variant_tag,
            # style_order=order,
            # ci=None, # save a lot time without error bars
            sort=False,
        )
        axes[ax_id].legend(framealpha=0.5, loc="upper left")
        axes[ax_id].set_title(variant_tag_name)
        # if FLAGS.max_x is not None:
        #     axes[ax_id].set_xlim(0, FLAGS.max_x)
        ax_id += 1

# set the rest subplots blank
while ax_id < rows * cols:
    axes[ax_id].set_visible(False)
    ax_id += 1

plt.tight_layout()
# plt.show()
# plt.close()
plt.savefig(
    os.path.join(
        FLAGS.merged_path,
        f"single_factor-{''.join(FLAGS.factors)}-window{FLAGS.window_size}.png",
    ),
    dpi=200,
    bbox_inches="tight",
)
plt.close()


###  3. draw the top curves


def get_run_down(dataframe, key, last_steps_ratio=FLAGS.last_steps_ratio):
    dataframe[key + auc_tag] = dataframe.groupby([merged_tag, trial_tag])[
        key
    ].transform(
        lambda x: x[int(last_steps_ratio * len(x)) :].mean()  # last few timesteps
    )
    tmp_df = dataframe.groupby([merged_tag, trial_tag]).tail(1)  # last timestep
    run_down = tmp_df.groupby([merged_tag])[
        key + auc_tag
    ].mean()  # avg auc for each instance
    run_down_std = tmp_df.groupby([merged_tag])[
        key + auc_tag
    ].std()  # NOTE: std over seeds (not env steps!)
    run_down_std.name += "_std"
    return run_down, run_down_std


topk_tags_dict = {}
run_down_df = []
for key in key_of_interests:
    # select top k of ours
    run_down_ours, run_down_ours_std = get_run_down(df_ours, key)
    run_down_ours = run_down_ours.nlargest(run_down_ours.shape[0])
    # keep all the others
    run_down_others, run_down_others_std = get_run_down(df_others, key)
    run_down_sorted = pd.concat([run_down_ours, run_down_others]).sort_values(
        ascending=False
    )
    run_down_std = pd.concat([run_down_ours_std, run_down_others_std])
    run_down_std_sorted = run_down_std[run_down_sorted.index]
    # join the sorted two series into one sorted dataframe
    run_down = pd.concat([run_down_sorted, run_down_std_sorted], axis=1)

    run_down_df.append(run_down)

    # only plot subset of top k for visualization
    # like involution :(
    pool = run_down.index.to_list()
    topk_tags_dict[key] = []
    for candidate in pool:
        if candidate in specialized_tags:  # others
            topk_tags_dict[key].append([candidate, "others"])
        else:
            topk_tags_dict[key].append([candidate, "ours"])


run_down_df = pd.concat(run_down_df, axis=1)
run_down_df.insert(0, merged_tag, run_down_df.index)  # insert to the 1st column
run_down_df = run_down_df.round(decimals=3)
print(run_down_df)

run_down_df.to_csv(
    os.path.join(
        FLAGS.merged_path,
        f"rundown-max_x{FLAGS.max_x}-last{FLAGS.last_steps_ratio}-window{FLAGS.window_size}.csv",
    ),
    index=False,
)


## draw the top curves
sns.set(font_scale=1.25)
num_plots = len(key_of_interests[1:])
cols = int(np.ceil(np.sqrt(num_plots)))
rows = int(np.ceil(num_plots / cols))

# seaborn plot
fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4))
axes = (
    axes.flatten() if isinstance(axes, np.ndarray) else [axes]
)  # make it as a flat list

ax_id = 0
for key in key_of_interests[1:]:
    # select topk merged_tags
    if FLAGS.best_variant is not None:
        plot_df_ours = df_ours[df_ours[merged_tag] == FLAGS.best_variant]
    else:
        plot_df_ours = df_ours.loc[
            df_ours[merged_tag].isin([name for name, _ in topk_tags_dict[key]])
        ].copy()
    plot_df_ours["method"] = ours_name

    # select other methods
    if FLAGS.other_methods is not None:
        if len(FLAGS.other_methods) == 1 and FLAGS.other_methods[0] == "No":
            plot_df_others = pd.DataFrame()
        else:
            plot_df_others = df_others.loc[
                df_others["method"].isin(FLAGS.other_methods)
            ]
    else:
        plot_df_others = df_others

    plot_df = pd.concat(
        [plot_df_ours, plot_df_others], ignore_index=True
    )  # return-auc is NaN for others

    sns.lineplot(
        ax=axes[ax_id],
        data=plot_df,
        x=x_tag,
        y=key,
        palette={
            key: color for key, (color, _) in curve_style_dict.items()
        },  # fix the palette
        dashes={key: dash for key, (_, dash) in curve_style_dict.items()},
        hue="method",
        style="method",
        # style_order=topk_tags_dict[key],
        # ci=None, # save a lot time without error bars
        sort=False,
    )
    for baseline_name, baseline_scalar in table_results[FLAGS.name][key].items():
        axes[ax_id].axhline(
            y=baseline_scalar,
            label=baseline_name,
            color=curve_style_dict[baseline_name][0],
        )

    axes[ax_id].legend(framealpha=0.5, loc="upper left")
    # if FLAGS.max_x is not None:
    #     axes[ax_id].set_xlim(0, FLAGS.max_x)
    ax_id += 1

plt.tight_layout()
# plt.show()
# plt.close()

fig_name = f"instance-{FLAGS.best_variant}-max_x{FLAGS.max_x}-last{FLAGS.last_steps_ratio}-window{FLAGS.window_size}"
if FLAGS.other_methods is not None:
    fig_name += f"-others{'_'.join(FLAGS.other_methods)}"

save_path = os.path.join(FLAGS.merged_path, fig_name + ".png")

plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()
