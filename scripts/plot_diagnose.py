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
flags.DEFINE_integer("window_size", 50, "window size of the plot")
# flags.DEFINE_integer("top_k", 5, "top k curves of the plot")
# flags.DEFINE_float("last_steps_ratio", 0.80, "for measureing top k")
flags.DEFINE_list("instances", None, "instances for diagnosis plots")
flags.FLAGS(sys.argv)
print(FLAGS.flags_into_string())

df = pd.read_csv(FLAGS.csv_path)

# extract the tags
x_tag = x_tag[1]
perf_tags = [y_tag[1] for y_tag in y_tags if y_tag[1] in df.columns]
# add diagnosis tags: # remove redundancy while keep order
q_rnn_tag = "q_rnn_grad_norm"
pi_rnn_tag = "pi_rnn_grad_norm"
rnn_tag = "rnn_grad_norm"
diagnosis_tags = [rnn_tag]  # NOTE: hard code
# list(dict.fromkeys(
#     [y_tag[1] for y_tag in diagnosis_tags if y_tag[1] in df.columns]
# ))
key_of_interests = perf_tags + diagnosis_tags
print(key_of_interests)

variant_tag_names = [
    variant_tag_name
    for variant_tag_name in variant_tag_names
    if variant_tag_name in df.columns
]

# smoothing
for key in perf_tags:
    df[key] = df.groupby([*variant_tag_names, trial_tag])[key].transform(
        lambda x: x.rolling(FLAGS.window_size, min_periods=1).mean()  # rolling mean
    )

## draw the top curves
sns.set(font_scale=1.5)
# create a merged tag
merged_tag = "instance"
df_ours = df.loc[df["method"] == "ours"]
# https://stackoverflow.com/questions/19377969/combine-two-columns-of-text-in-pandas-dataframe
df_ours[merged_tag] = df_ours[variant_tag_names].astype(str).agg("-".join, axis=1)
assert len(FLAGS.instances) == 2  # for palette
plot_df = df_ours.loc[df_ours[merged_tag].isin(FLAGS.instances)]

num_plots = len(key_of_interests)
cols = int(np.ceil(np.sqrt(num_plots)))
rows = int(np.ceil(num_plots / cols))

# seaborn plot
fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4))
axes = (
    axes.flatten() if isinstance(axes, np.ndarray) else [axes]
)  # make it as a flat list

ax_id = 0
for key in perf_tags:

    sns.lineplot(
        ax=axes[ax_id],
        data=plot_df,
        x=x_tag,
        y=key,
        palette=("blue", "red"),
        hue=merged_tag,
        # hue_order=topk_tags_dict[key], # sorted order
        # style=merged_tag,
        # style_order=topk_tags_dict[key],
        # ci=None, # save a lot time without error bars
        sort=False,
    )
    axes[ax_id].legend(framealpha=0.5)
    if FLAGS.max_x is not None:
        axes[ax_id].set_xlim(0, FLAGS.max_x)
    ax_id += 1

## now process and draw diagnosis_tags

plot_df_q = plot_df[plot_df[q_rnn_tag].notna()].copy()
plot_df_q[rnn_tag] = plot_df_q[q_rnn_tag]
plot_df_q[merged_tag] = plot_df_q[merged_tag] + ":critic"

plot_df_pi = plot_df[plot_df[pi_rnn_tag].notna()].copy()
plot_df_pi[rnn_tag] = plot_df_pi[pi_rnn_tag]
plot_df_pi[merged_tag] = plot_df_pi[merged_tag] + ":actor"

plot_diagnosis_df = plot_df.loc[plot_df[rnn_tag].notna()].copy()
plot_diagnosis_df = pd.concat(
    [plot_diagnosis_df, plot_df_q, plot_df_pi], ignore_index=True
)


for key in diagnosis_tags:
    plot_diagnosis_df[key] = plot_diagnosis_df[key].clip(lower=1e-5)

    sns.lineplot(
        ax=axes[ax_id],
        data=plot_diagnosis_df,
        x=x_tag,
        y=key,
        # sort in order from high to low, thus red is shared, the blues is separate
        palette=("red", "blue", "blue"),
        hue=merged_tag,
        # hue_order=topk_tags_dict[key], # sorted order
        style=merged_tag,
        # style_order=topk_tags_dict[key],
        # ci=None, # save a lot time without error bars
        sort=False,
    )
    axes[ax_id].legend(framealpha=0.5)
    if FLAGS.max_x is not None:
        axes[ax_id].set_xlim(0, FLAGS.max_x)
    axes[ax_id].set_yscale("log")
    ax_id += 1


plt.tight_layout()
# plt.show()
# plt.close()
plt.savefig(
    os.path.join(*FLAGS.csv_path.split("/")[:-1], "diagnosis.png"),
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.1,
)
plt.close()
