import torch.nn as nn
import sys
sys.path.append("modules")
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
import torch


class WeightDropBiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size,  numclass = 0,dropouti = 0.05, wdrop = 0.2, dropouto = 0.05):
        super(WeightDropBiLSTM, self).__init__()
        wdrop = 0.05
        self.rnn1 = nn.LSTM(input_size, hidden_size, bidirectional=True,)
        self.linear_rnn = nn.Linear(hidden_size * 2, hidden_size)
        self.rnn2 = nn.LSTM(hidden_size , hidden_size, bidirectional=True,)
        self.linear = nn.Linear(hidden_size * 2, numclass)
        self.lockdrop = LockedDropout()
        self.weight_drop1 = WeightDrop(self.rnn1, ['weight_hh_l0'], dropout=wdrop)
        self.weight_drop2 = WeightDrop(self.rnn2, ['weight_hh_l0'], dropout=wdrop)
        self.dropouti = dropouti
        self.dropouto = dropouto
        self.dropouti = 0.05
        self.dropouto = 0.05
        initrange = 1
        self.linear_rnn.weight.data.uniform_(-initrange, initrange)
        self.linear_rnn.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        input = input.permute(1, 0, 2)
        output, _ = self.weight_drop1(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.lockdrop(output, self.dropouti)
        linear1 = self.linear_rnn(output)
        output, _ = self.weight_drop2(linear1)
        output = self.lockdrop(output, self.dropouto)
        output = self.linear(output)  # batch_size x T x output_size
        return output.permute(1, 0, 2)
