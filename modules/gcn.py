
import math

import torch

import torch.nn as nn


class GraphConvolution(nn.modules.module.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, batch_size, len_sequence, in_features, out_features, bias=False, scale_factor = 0., dropout = 0.0, isnormalize = False):
        super(GraphConvolution, self).__init__()
        self.batch_size = batch_size
        self.in_features = in_features 
        self.out_features = out_features
        self.LinearInput = nn.Linear(in_features, in_features)
        self.CosineSimilarity = nn.CosineSimilarity(dim=-2, eps=1e-8)
        self.len_sequence = len_sequence
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.isnormalize = isnormalize
        self.distance_matrix = self.get_distance_matrix( len_sequence, scale_factor).to(self.device)
        self.eye_matrix = torch.eye(len_sequence)
        # self.weight = torch.nn.parameter.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        self.OutputLayers = nn.Sequential(
            nn.Linear(in_features, out_features, bias = bias),
            nn.BatchNorm1d(len_sequence),
            torch.nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

        def reset_parameters(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                # m.bias.data.fill_(0.0)
        self.OutputLayers.apply(reset_parameters)
        # if bias:
        #     self.bias = torch.nn.parameter.Parameter(torch.FloatTensor(out_features)).to(device)
        # else:
        #     self.register_parameter('bias', None)
        

    def get_distance_matrix(self, len_sequence, scale_factor):
        tmp = torch.arange(float(len_sequence)).repeat(len_sequence, 1)
        tmp = 1 / (1 + torch.exp(torch.abs(tmp-torch.transpose(tmp, 0, 1))-scale_factor))
        tmp[tmp < 0.25]  = 0
        return tmp.unsqueeze(0)


    # def normalize_pygcn(adjacency_maxtrix):
    #     """ normalize adjacency matrix with normalization-trick. This variant
    #     is proposed in https://github.com/tkipf/pygcn .
    #     Refer https://github.com/tkipf/pygcn/issues/11 for the author's comment.
    #     Arguments:
    #         a (scipy.sparse.coo_matrix): Unnormalied adjacency matrix
    #     Returns:
    #         scipy.sparse.coo_matrix: Normalized adjacency matrix
    #     """
    #     # no need to add identity matrix because self connection has already been added
    #     # a += sp.eye(a.shape[0])
    #     rowsum = np.array(adjacency_maxtrix.sum(1))
    #     rowsum_inv = np.power(rowsum, -1).flatten()
    #     rowsum_inv[np.isinf(rowsum_inv)] = 0.
    #     # ~D in the GCN paper
    #     d_tilde = sp.diags(rowsum_inv)
    #     return d_tilde.dot(a)
    def normalize_pygcn(self, adjacency_maxtrix, net):
        adjacency_maxtrix = adjacency_maxtrix + torch.eye(self.len_sequence).to(self.device)
        rowsum = torch.sum(adjacency_maxtrix,2)
        rowsum_inv = torch.pow(rowsum, -1)
        rowsum_inv[torch.isinf(rowsum_inv)] = 0.
        d_tilde = torch.diag_embed(rowsum_inv, 0)
        return  torch.einsum('bij,bjk,bkl->bil',d_tilde,adjacency_maxtrix,net)
        
        

    def cosine_pairwise(self,x):
        x = x.permute((1, 2, 0))
        cos_sim_pairwise = self.CosineSimilarity(x, x.unsqueeze(1))
        cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
        return cos_sim_pairwise


    def forward(self, input):
        net = input
        c = self.LinearInput(net)
        similarity_maxtrix = self.cosine_pairwise(c)
        adjacency_maxtrix = similarity_maxtrix * self.distance_matrix 
        if self.isnormalize :
            net = self.normalize_pygcn(adjacency_maxtrix, net)
        else :
            net = torch.einsum('ijk,ikl->ijl',adjacency_maxtrix, net)
        net = self.OutputLayers(net)
        return net

    def __repr__(self):
        return self.__class__.__name__ + ' ('                + str(self.in_features) + ' -> '                + str(self.out_features) + ')'






