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
