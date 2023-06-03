"""Dual Attention Graph Knowledge Tracing modules v2"""
import numpy as np
import torch as th
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import RelGraphConv
from .egatconv import EGATConv

import copy


class SAGKT(nn.Module):

    def __init__(self, in_nfeats, in_efeats, latent_dim,
                 num_heads=4, edge_dropout=0.2, subgraph_embedding_from='user_item'):
                 
        super(SAGKT, self).__init__()
        self.edge_dropout = edge_dropout
        self.in_nfeats = in_nfeats
        self.subgraph_embedding_from = subgraph_embedding_from
        self.elu = nn.ELU()
        self.leakyrelu = th.nn.LeakyReLU()
        self.local_egats = th.nn.ModuleList()
        latent_dim = copy.deepcopy(latent_dim)
        latent_dim.insert(0, in_nfeats)

        for i in range(0, len(latent_dim)-1):
            self.local_egats.append(EGATConv(in_node_feats=latent_dim[i],
                                            in_edge_feats=in_efeats,
                                            out_node_feats=latent_dim[i+1],
                                            out_edge_feats=8,
                                            num_heads=num_heads))

        self.lin1 = nn.Linear(2*sum(latent_dim[1:]), 128)
        self.dropout1 = nn.Dropout(0.5)
        self.lin2 = nn.Linear(128, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def get_parameters(self):
        parameters_dict = {}
        n, p  = self.lin1.named_parameters()
        parameters_dict[n] = p
        n, p  = self.lin2.named_parameters()
        parameters_dict[n] = p
        return parameters_dict


    def forward(self, graph):
        """ graph : subgraph batch """
        graph.edata['norm'] = graph.edata['edge_mask']
        node_x = graph.ndata['x'].float()

        states = []


        x = node_x # original
        e = graph.edata['efeat']

        for local_egat in self.local_egats:
            x, _ = local_egat(graph=graph, nfeats=x, efeats=e,)
            x = x.sum(dim=1)
            x = self.elu(x)
            states.append(x)

        states = th.cat(states, 1)
        x = self._get_subgraph_embedding(states, graph)
        x = th.relu(self.lin1(x))
        x = self.dropout1(x)
        x = self.lin2(x)
        x = th.sigmoid(x)

        return x[:, 0]

    def _get_subgraph_embedding(self, states, graph):
        # get user, item idx --> vector
        users = graph.ndata['ntype'] == 0
        items = graph.ndata['ntype'] == 1
        subgraphs = graph.ndata['ntype'] == -1

        if self.subgraph_embedding_from == 'center':
            x = states[subgraphs]
        elif self.subgraph_embedding_from == 'user_item':
            x = th.cat([states[users], states[items]], 1)
        else:
            NotImplementedError()

        return x
