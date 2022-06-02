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
flags.DEFINE_list("factors", None, "selected single factors")
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
if FLAGS.factors is None:
    FLAGS.factors = variant_tag_names
assert set(FLAGS.factors) <= set(variant_tag_names)  # subset check

if FLAGS.max_x is not None:
    df = df.loc[df[x_tag] <= FLAGS.max_x]  # set max_x

# smoothing
for key in key_of_interests:
    df[key] = df.groupby([*variant_tag_names, trial_tag])[key].transform(
        lambda x: x.rolling(FLAGS.window_size, min_periods=1).mean()  # rolling mean
    )

# make a square-like plot
num_plots = len(key_of_interests) * max(1, len(FLAGS.factors))
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
ax_id = 0
for key in key_of_interests:
    for variant_tag_name in FLAGS.factors:
        sns.lineplot(
            ax=axes[ax_id],
            data=df_ours,
            x=x_tag,
            y=key,
            palette=variant_colors[variant_tag_name]
            if variant_tag_name != "Len"
            else None,
            hue=variant_tag_name,
            # hue_order=order,
            # style=variant_tag,
            # style_order=order,
            # ci=None, # save a lot time without error bars
            sort=False,
        )
        # the follow command will remove the legend hue, why?
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
        *FLAGS.csv_path.split("/")[:-1],
        f"single_factor-{''.join(FLAGS.factors)}-window{FLAGS.window_size}.png",
    ),
    dpi=200,
    bbox_inches="tight",
)
plt.close()
