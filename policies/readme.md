# Our implementation of recurrent model-free RL

In `models/` folder, we provide 4 different model-free RL architectures:
- `policy_mlp.py`: Markov Actor and Markov Critic
- `policy_rnn_mlp.py`: Markov Actor and Recurrent Critic
- `policy_rnn_shared.py`: Recurrent Actor and Recurrent Critic with shared RNN
- `policy_rnn.py`: Recurrent Actor and Recurrent Critic with separate RNNs
