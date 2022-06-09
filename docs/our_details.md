# Details of Our Implementation of Recurrent Model-Free RL

## Configuration of Decision Factors

The **decision factors** discussed in the method section of our paper, can be found in each yaml file (take [`configs/meta/point_robot/rnn.yml`](../configs/meta/point_robot/rnn.yml) as an example, which refers to meta RL environment Semi-Circle)

- Arch: `policy: separate: <True|False>` 
    - Recommmend to set it as `True`
- Encoder: `policy: seq_model: <lstm|gru>`
    - We also support `mlp` option for Markovian policies
- RL: `policy: algo_name: <td3|sac|sacd>`
- Len: `train: sampled_seq_len: <5|64|any positive integer>`
- Inputs: we use the embedding size to control the inputs. 
    - To use past observations, simply set `policy: observ_embedding_size` as a positive integer
    - To use past actions, set `policy: action_embedding_size` as a positive integer
    - To use past rewards, set `policy: reward_embedding_size` as a positive integer
    - Otherwise, if you want to disable any of them, set the corresponding embedding size as 0.

## Other Important Training Hyperparameters

- The total number of environment steps: it is controlled by `train: num_iters: <int>` 
- The update frequency of RL algorithm w.r.t. the environment step: it is controlled by `train: num_updates_per_iter: <int|float>`.
    - If you use `int`, it directly set the number of gradient steps
    - If use `float`, it will set the number of gradient steps computed by the relative ratio of environment steps
    - This hparam is crucial to computation speed and sample efficiency
- Enable tensorboard: set `eval: log_tensorboard: True`
- GPU usage: set `cuda: <int>` as the CUDA device number, if `-1` then disable GPU usage to purely use CPU.

## Code-Level Details
- Replay buffer: check [`SeqReplayBuffer`](../buffers/seq_replay_buffer.py) to see the implementation of 2-dim replay buffer that supports sequence storage and sampling.
- Our implemention: check [`ModelFreeOffPolicy_Separate_RNN`](../policies/models/policy_rnn.py) for (separate) recurrent model-free RL architecture
