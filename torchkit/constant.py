import torch.nn as nn

TD3_name = "td3"
SAC_name = "sac"
SACD_name = "sacd"

LSTM_name = "lstm"
GRU_name = "gru"
RNNs = {
    LSTM_name: nn.LSTM,
    GRU_name: nn.GRU,
}

relu_name = "relu"
elu_name = "elu"
ACTIVATIONS = {
    relu_name: nn.ReLU,
    elu_name: nn.ELU,
}
