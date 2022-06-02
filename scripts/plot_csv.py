"""
Plot selective variant of our method and all the compared methods
"""
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
flags.DEFINE_string("csv_path", None, "csv path")
flags.DEFINE_integer("max_x", None, "max value of x")
flags.DEFINE_integer("window_size", 20, "window size of the plot")
flags.DEFINE_float("last_steps_ratio", 0.80, "for measureing top k")
flags.DEFINE_float("normalizer", None, "for normalizing the return in the plots")
flags.DEFINE_string("best_variant", None, "best variant of our method if given")
flags.DEFINE_list("other_methods", None, "selective other methods to show if given")
flags.DEFINE_string("name", None, "env name for adding constant horizontal lines")
flags.DEFINE_string("loc", "upper left", "legend")
flags.FLAGS(sys.argv)
print(FLAGS.flags_into_string())

df = pd.read_csv(FLAGS.csv_path)

# extract the tags
x_tag = x_tag[1]
key_of_interests = [y_tag[1] for y_tag in y_tags if y_tag[1] in df.columns]
variant_tag_names = [
    variant_tag_name
    for variant_tag_name in variant_tag_names
    if variant_tag_name in df.columns
]

if FLAGS.max_x is not None:
    df = df.loc[df[x_tag] <= FLAGS.max_x]  # set max_x

# smoothing
for key in key_of_interests:
    if FLAGS.normalizer is not None:
        df[key] /= FLAGS.normalizer

    df[key] = df.groupby(
        [method_tag, *variant_tag_names, trial_tag]  # fix BUG: add method_tag
    )[key].transform(
        lambda x: x.rolling(FLAGS.window_size, min_periods=1).mean()  # rolling mean
    )

df_ours = df.loc[df[method_tag] == "ours"]
df_others = df.loc[df[method_tag] != "ours"]

# https://stackoverflow.com/questions/19377969/combine-two-columns-of-text-in-pandas-dataframe
df_ours[merged_tag] = df_ours[variant_tag_names].astype(str).agg("-".join, axis=1)
df_others[merged_tag] = df_others[method_tag]


def get_run_down(dataframe, key, last_steps_ratio=FLAGS.last_steps_ratio):
    dataframe[key + auc_tag] = dataframe.groupby([merged_tag, trial_tag])[
        key
    ].transform(
        lambda x: x[
            int(last_steps_ratio * len(x)) :
        ].mean()  # last few timesteps of each trial
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
    run_down_ours = run_down_ours.nlargest(run_down_ours.shape[0])  # rank

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
        *FLAGS.csv_path.split("/")[:-1],
        f"rundown-{'-'.join(key_of_interests)}"
        + f"-max_x{FLAGS.max_x}-last{FLAGS.last_steps_ratio}-window{FLAGS.window_size}.csv",
    ),
    index=False,
)


## draw the top curves
sns.set(font_scale=1.25)
num_plots = len(key_of_interests)
cols = int(np.ceil(np.sqrt(num_plots)))
rows = int(np.ceil(num_plots / cols))

# seaborn plot
fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4))
axes = (
    axes.flatten() if isinstance(axes, np.ndarray) else [axes]
)  # make it as a flat list

ax_id = 0
for key in key_of_interests:
    # select our variant
    if FLAGS.best_variant is not None:
        plot_df_ours = df_ours[df_ours[merged_tag] == FLAGS.best_variant]
        plot_df_ours[merged_tag] = ours_name
    else:
        plot_df_ours = pd.DataFrame()

    # select other methods
    if FLAGS.other_methods is not None:
        plot_df_others = df_others.loc[df_others[method_tag].isin(FLAGS.other_methods)]
    else:
        plot_df_others = df_others

    plot_df = pd.concat(
        [plot_df_ours, plot_df_others], ignore_index=True
    )  # return-auc is NaN for others

    order = []
    for name, _ in topk_tags_dict[key]:
        if name in plot_df[merged_tag].values:  # for others
            order.append(name)
        elif name == FLAGS.best_variant:  # our name is raw
            order.append(ours_name)

    sns.lineplot(
        ax=axes[ax_id],
        data=plot_df,
        x=x_tag,
        y=key,
        palette={
            key: color for key, (color, _) in curve_style_dict.items()
        },  # fix the palette
        dashes={key: dash for key, (_, dash) in curve_style_dict.items()},
        hue=merged_tag,
        hue_order=order,  # sorted order
        style=merged_tag,
        # style_order=topk_tags_dict[key],
        # ci=None, # save a lot time without error bars
        sort=False,
    )
    if FLAGS.normalizer is not None:
        axes[ax_id].set_ylabel("Normalized " + key)

    if FLAGS.name is not None:
        for baseline_name, baseline_scalar in table_results[FLAGS.name][key].items():
            axes[ax_id].axhline(
                y=baseline_scalar,
                label=baseline_name,
                color=curve_style_dict[baseline_name][0],
                # marker='o' if baseline_name == "Oracle" else None, markersize=15,
            )

    axes[ax_id].legend(framealpha=0.5, loc=FLAGS.loc)  # fix location
    # axes[ax_id].set_ylim(None, -50)
    if FLAGS.max_x is not None:
        axes[ax_id].set_xlim(0, FLAGS.max_x)
    ax_id += 1

plt.tight_layout()
# plt.show()
# plt.close()

fig_name = f"{'-'.join(key_of_interests)}"
fig_name += f"-instance-{FLAGS.best_variant}-max_x{FLAGS.max_x}-last{FLAGS.last_steps_ratio}-window{FLAGS.window_size}"
if FLAGS.other_methods is not None:
    fig_name += f"-others{'_'.join(FLAGS.other_methods)}"

save_path = os.path.join(*FLAGS.csv_path.split("/")[:-1], fig_name + ".png")

plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()
