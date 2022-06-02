import pandas as pd
import numpy as np
import os, sys
import glob
from scripts.constants import *
from absl import flags

FLAGS = flags.FLAGS

## Hparams
flags.DEFINE_string("base_path", None, "Base path")
flags.DEFINE_integer("start_x", 0, "start step of record")
flags.DEFINE_integer("interval_x", 1, "interval of record")
flags.DEFINE_integer("max_episode_len", 1000, "max episode length")
flags.DEFINE_string("output_csv", "final.csv", "final csv name")
flags.FLAGS(sys.argv)
print(FLAGS.flags_into_string())


# standard fixed-interval env steps
start_x = FLAGS.start_x
interval_x = FLAGS.interval_x

## Find all csvs
file_paths = glob.glob(os.path.join(FLAGS.base_path, "**/progress.csv"), recursive=True)
print(file_paths)

results = []

for path in file_paths:
    if "oracle" not in FLAGS.base_path and "oracle/" in path:
        continue
    df = pd.read_csv(path)
    result = pd.DataFrame()  # to be filled

    # 1. record metrics data
    #   (after interpolation to stardard fixed-interval steps)

    ## 1D interp on y data
    for y_tag in y_tags:
        if y_tag[0] not in df.columns:
            continue

        # use y_tag to get valid entries, not x_tag
        # assume all y_tags aligned
        valid_entries = df[y_tag[0]].notna()  # remove NaN
        y_raw = df[y_tag[0]][valid_entries]
        x_raw = df[x_tag[0]][valid_entries]

        ## interpolate
        end_x = x_raw.max()  # fix bug!
        x_interp = np.arange(start_x, end_x + 1, interval_x)
        y_interp = np.interp(x_interp, x_raw, y_raw)
        result[x_tag[1]] = x_interp
        result[y_tag[1]] = np.around(y_interp, decimals=2)

    diagnosis_indices = valid_entries[valid_entries == True].index - 1
    for diagnosis_tag in diagnosis_tags:
        if diagnosis_tag[0] not in df.columns:
            continue
        valid_entries_attempt = df[x_tag[0]].notna() & df[diagnosis_tag[0]].notna()
        assert valid_entries_attempt.any() == False  # logging issue

        diagnosis_y_raw = df.iloc[diagnosis_indices][diagnosis_tag[0]]
        assert len(diagnosis_y_raw) == len(x_raw)
        ## interpolate using x_raw as proxy
        diagnosis_y_interp = np.interp(x_interp, x_raw, diagnosis_y_raw)
        result[diagnosis_tag[1]] = np.around(diagnosis_y_interp, decimals=2)

    # 2. record meta-data
    #   simply using df[tag] = str will broadcast

    ## parse trial_str to get variant tags
    trial_str_list = list(filter(None, path.replace(FLAGS.base_path, "").split("/")))
    trial_str = "/".join(trial_str_list)
    print(trial_str)

    trial_time_str = trial_str_list[-2]  # -1 is "progress.csv"
    result[trial_tag] = trial_time_str

    ## hparams
    if any(name in trial_str for name in baseline_names):
        # belong to our baselines
        result[method_tag] = "ours"
        variant_tags = get_variant_tags(trial_str, FLAGS.max_episode_len)
        for k, v in variant_tags.items():
            result[k] = v
    else:
        # specialized or other methods
        specialized_name = trial_str_list[0]
        assert specialized_name in specialized_tags
        result[method_tag] = specialized_name
        for k, v in specialized_tags[specialized_name].items():
            if k in variant_tag_names:
                result[k] = v

    results.append(result)

results = pd.concat(results)
os.makedirs(FLAGS.base_path.replace("logs", "data"), exist_ok=True)
results.to_csv(
    os.path.join(FLAGS.base_path.replace("logs", "data"), FLAGS.output_csv), index=False
)
