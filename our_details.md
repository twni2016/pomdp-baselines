## Details of Our Implementation of Recurrent Model-Free RL

### Configuration of Decision Factors

The **decision factors** discussed in the method section in our paper, can be found in each yaml file (take `configs/meta/point_robot/rnn.yml` as example, which refers to meta RL environment Semi-Circle)

- Arch: `policy: separate: <True|False>` 
- Encoder: `policy: arch: <lstm|gru>`
- RL: `policy: algo: <td3|sac>`
- Len: `train: sampled_seq_len: <5|64|any positive integer>`
- Inputs: we use the embedding size to control the inputs. 
    - To use past observations, simply set `policy: state_embedding_size` a positive integer
    - To use past actions, set `policy: action_embedding_size` a positive integer
    - To use past rewards, set `policy: reward_embedding_size` a positive integer
    - Otherwise, if you want to disable any of them, set the corresponding embedding size as 0.

### Other Important Training Hyperparameters

- The total number of environment steps: it is controlled by `train: num_iters: <int>` 
- The update frequency of RL algorithm w.r.t. the environment step: it is controlled by `train: num_updates_per_iter: <int|float>`.
    - If you use `int`, it directly set the number of gradient steps
    - If use `float`, it will set the the number of gradient steps computed by the relative ratio of environment steps
    - This hparam is crucial to computation speed.
- Enable tensorboard: set `eval: log_tensorboard: True`
- GPU usage: set `cuda: <int>` as the CUDA device number, if `-1` then disable GPU usage.

### Code-level Details
- Replay buffer: check `buffers/seq_replay_buffer.py` to see the implementation of 2-dim replay buffer that supports sequence storage and sampling.
- Our implemention: check the class `ModelFreeOffPolicy_Separate_RNN` in `policies/models/policy_rnn.py`


### Final Results that Generate the Bar Charts in the Paper
Please download the results `data.zip` from the [google drive](https://drive.google.com/file/d/18l9Y4N8zPRdGBnx8oSELiQcoReF7V4wP/view?usp=sharing) and decompress into `data` folder.

In `data/<subarea>` folder, we shared the final results that generate the bar charts in the paper. 

- `data/<subarea>/rank*.csv` show the ranking of each variant in our implemention by the performance metric averaged across the environments in each subarea. For example, the instance `td3-gru-64-oa-separate` appears first in the `data/pomdp/rank_return-max_x1500000.csv`, thus it is the best variant.

- `data/<subarea>/<env_name>/run_down*.csv` show the final results of each variant in our implemention and the compared methods in each environment

We also provide all the learning curves results in `data/<subarea>/<env_name>/final.csv` for reproducibility and usage in future work.

