import torch.nn as nn


class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.lstm = nn.LSTM(100, 128, 2, bidirectional=True, batch_first=True,
                            dropout=0.05)  # 词嵌入维度为100,,lstm层的隐藏单元为128，lstm层的层数为2
        self.fc = nn.Linear(39 * 256, 6)
        self.init_weights()

    def init_weights(self):
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)

        nn.init.orthogonal_(self.lstm.weight_ih_l1)
        nn.init.orthogonal_(self.lstm.weight_hh_l1)

        nn.init.zeros_(self.lstm.bias_ih_l1)
        nn.init.zeros_(self.lstm.bias_hh_l1)

        self.fc.bias.data.fill_(0)
        self.fc.weight.data.normal_(0, 0.01)

    def forward(self, x):  # x [batch_size, seq_len, embeding]=[16,39,100]
        out, _ = self.lstm(x)
        # out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        out1 = out.contiguous().view(128, -1)
        out2 = self.fc(out1)
        return out2
