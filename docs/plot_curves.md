# Draw and Download the Learning Curves

## Draw Learning Curves

First, assume we have run some experiments and got its learning curve data in `logs/` folder. Then in <local_path>
```bash
mkdir results
cp -r logs/ results/ # the logs are copied to results/logs
```

Then, we have to merge the learning curve data (many csvs) into one `final.csv` saved in `results/data` folder:
```bash
python scripts/merge_csv.py --base_path results/logs/<subarea>/<env_name>
```

After that, we can draw the learning curves `instance*.png` of selected methods and get the `rundown*.csv` (summary) of all the methods:
```bash
python scripts/plot_csv.py --csv_path results/data/<subarea>/<env_name>/final.csv  \
    --best_variant <recurrent model free RL variant name> --other_methods <the method names we want to show in the plot>
```

We can also draw single factor analysis plot of recurrent model-free RL:
```bash
python scripts/plot_single_factor.py --csv_path results/data/<subarea>/<env_name>/final.csv \
    --factors <RL,Encoder,Len,Inputs>
```

After collecting all the log data for one benchmark, we can generate the ranking on average normalized return `rank*.csv`:
```bash
python scripts/show_ranking.py --base_path results/data/<subarea>
```

For adding new methods or environments, please register them in [constants.py](../scripts/constants.py).

**Finally, we provide the running script to generate all the plots in the paper in [eval.sh](../scripts/eval.sh). Please check that script for details on the optional arguments.**


## Download Final Results that Generate the Learning Curves in the Paper

Please download the results `data.zip` from the [google drive](https://drive.google.com/file/d/1dfulN8acol-qaNR2h4PDpIaWBg9Ck4pY/view?usp=sharing) and decompress into `results/` folder.

- `results/data/<subarea>/rank*.csv` show the ranking of each variant in our implementation by the performance metric averaged across the environments in each subarea. For example, the instance `td3-gru-64-oa-separate` appears first in the `results/data/pomdp/rank_return-max_x1500000.csv`, thus it is the best variant.

- `results/data/<subarea>/<env_name>/rundown*.csv` show the final results of each variant in our implemention and the compared methods in each environment

- `results/data/<subarea>/<env_name>/final.csv` show the learning curve data

**Please run [eval.sh](../scripts/eval.sh) to generate all the plots in the paper** (comment the `merge_csv.py` commands as we already have `final.csv` files).

