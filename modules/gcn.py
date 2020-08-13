import math

import torch

from torch.nn as nn


class GraphConvolution(nn.modules.module.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, batch_size, len_sequence, in_features, out_features, bias=False, scale_factor = 0):
        super(GraphConvolution, self).__init__()
        self.batch_size = batch_size
        self.in_features = in_features # TxK
        self.out_features = out_features #TxK
        self.Linear = nn.Linear(in_features, in_features)
        self.CosineSimilarity = nn.CosineSimilarity(dim=-2, eps=1e-8)
        self.len_sequence = len_sequence
        self.distance_matrix = self.get_distance_matrix(batch_size, len_sequence, scale_factor)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def get_distance_matrix(self, batch_size, len_sequence, scale_factor):
        tmp = torch.arange(float(len_sequence)).repeat(len_sequence, 1)
        return (1 / (1 + torch.exp(torch.abs(tmp-torch.transpose(tmp, 0, 1))-scale_factor))).unsqueeze(0).repeat(batch_size,1,1)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def cosine_pairwise(x):
        x = x.permute((1, 2, 0))
        cos_sim_pairwise = self.CosineSimilarity(x, x.unsqueeze(1))
        cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
        return cos_sim_pairwise


    def forward(self, input):
        c = self.Linear(input)
        similarity_maxtrix = self.cosine_pairwise(c)
        adjacency_maxtrix = self.distance_matrix * similarity_maxtrix
        support = torch.mm(input, self.weight)
        output = torch.spmm(adjacency_maxtrix, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'