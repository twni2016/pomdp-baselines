import torch.nn as nn

LSTM_name = "lstm"
GRU_name = "gru"
RNNs = {
    LSTM_name: nn.LSTM,
    GRU_name: nn.GRU,
}
