import pandas as pd
import numpy as np
import os, sys
import glob
from scripts.constants import *
from absl import flags

FLAGS = flags.FLAGS

## Hparams
flags.DEFINE_string("base_path", None, "base dir of rundown*.csv path")
flags.DEFINE_string("max_x", None, "rundown_max_x*.csv")
flags.DEFINE_string("y_tag", "return", "base dir of csv path")
flags.FLAGS(sys.argv)
print(FLAGS.flags_into_string())

## Find all csvs

if FLAGS.max_x is None:
    run_down_regex = "**/rundown*.csv"
else:
    run_down_regex = f"**/rundown-max_x{FLAGS.max_x}-*.csv"
file_paths = glob.glob(os.path.join(FLAGS.base_path, run_down_regex), recursive=True)
file_paths = sorted(file_paths)
print(file_paths)

method_tag = "Method"
env_tag = "Environment"
y_tag = FLAGS.y_tag
rank_tag = "rank"
results = []

for path in file_paths:
    df = pd.read_csv(path)

    # normalized by the max
    df[rank_tag] = df[y_tag + auc_tag] / df[y_tag + auc_tag].max()
    df[env_tag] = get_env_tag(path)
    results.append(df)

df = pd.concat(results, axis=0, ignore_index=True)

# average the rank
df_rank = df.groupby([merged_tag])[[rank_tag]].mean()
df_rank = df_rank.sort_values(by=rank_tag, ascending=False)
df_rank = df_rank.round(4)

rank_path = f"rank_{y_tag}"
if FLAGS.max_x is not None:
    rank_path += f"-max_x{FLAGS.max_x}"
df_rank.to_csv(os.path.join(FLAGS.base_path, rank_path + ".csv"))
