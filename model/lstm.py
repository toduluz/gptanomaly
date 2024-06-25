import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Optional

class LogRobust(nn.Module):
    def __init__(self,
                 embedding_dim: int = 300,
                 hidden_size: int = 100,
                 num_layers: int = 2,
                 is_bilstm: bool = True,
                 n_class: int = 2,
                 dropout: float = 0.5,
    ):
        super(LogRobust, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=is_bilstm,
                            dropout=dropout)
        self.num_directions = 2 if is_bilstm else 1
        self.fc = nn.Linear(hidden_size * self.num_directions, n_class)

        self.attention_size = self.hidden_size
        self.w_omega = Variable(
            torch.zeros(self.hidden_size * self.num_directions, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))

    def attention_net(self, lstm_output, sequence_length, device='cpu'):
        output_reshape = torch.Tensor.reshape(lstm_output,
                                              [-1, self.hidden_size * self.num_directions])

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega.to(device)))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega.to(device), [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, sequence_length, 1])
        state = lstm_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, inp, device='cpu'):
        sequence_length = inp.size(1)
        out, _ = self.lstm(inp)
        out = self.attention_net(out, sequence_length, device)
        logits = self.fc(out)
        return logits

    