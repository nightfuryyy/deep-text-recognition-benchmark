
import torch.nn as nn
import torch

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, is_last_blstm = False, numclass = 0):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.is_last_blstm = is_last_blstm
        if is_last_blstm :
            self.linear = nn.Linear(hidden_size * 2 + input_size, numclass)
        

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        if self.is_last_blstm :
            output = self.linear(torch.cat((recurrent,input), dim = 2))  # batch_size x T x output_size
        else :
            output = self.linear(recurrent)
        return output
    
