"""Dual Attention Graph Knowledge Tracing modules v2"""
import numpy as np
import torch as th
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import RelGraphConv
from .egatconv import EGATConv
from .gat_v2 import GraphLevelAttention

import copy


class DAGKT(nn.Module):

    def __init__(self, in_nfeats, in_efeats, latent_dim,
                 num_heads=4, edge_dropout=0.2, subgraph_embedding_from='center'):
                 
        super(DAGKT, self).__init__()
        self.edge_dropout = edge_dropout
        self.in_nfeats = in_nfeats
        self.subgraph_embedding_from = subgraph_embedding_from
        self.elu = nn.ELU()
        self.leakyrelu = th.nn.LeakyReLU()
        self.local_egats = th.nn.ModuleList()
        self.global_egats = th.nn.ModuleList()
        latent_dim = copy.deepcopy(latent_dim)
        latent_dim.insert(0, in_nfeats)

        for i in range(0, len(latent_dim)-1):
            self.local_egats.append(RelGraphConv(latent_dim[i], latent_dim[i+1], 3, num_bases=2, self_loop=True,))
            self.global_egats.append(RelGraphConv(latent_dim[i+1], latent_dim[i+1], 3, num_bases=2, self_loop=True,))

        self.lin1 = nn.Linear(sum(latent_dim[1:])*1, 128)
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
        graph = edge_drop(graph, self.edge_dropout, self.training)

        graph.edata['norm'] = graph.edata['edge_mask']
        node_x = graph.ndata['x'].float()



        x = node_x # original
        e = graph.edata['etype']
        mask = graph.edata['edge_mask']
        mask2 = graph.edata['edge_mask2']
        e2 = graph.edata['efeat2']
        subg_states = []
        ui_states = []

        for local_egat, global_egat in zip(self.local_egats, self.global_egats):
            # print(graph.edges()[0].shape, e.shape)
            x = local_egat(graph, x, e, norm=mask.unsqueeze(1))
            x = self.elu(x)
            ui_states.append(x)

            x = global_egat(graph, x, e, norm=mask2.unsqueeze(1))
            x = self.leakyrelu(x)
            graph.ndata['x'] = x
            subg_states.append(x)

        ui_states = th.cat(ui_states, 1)
        subg_states = th.cat(subg_states, 1)
        x = self._get_subgraph_embedding(ui_states, subg_states, graph)
        x = th.relu(self.lin1(x))
        x = self.dropout1(x)
        x = self.lin2(x)
        x = th.sigmoid(x)

        return x[:, 0]

    def _get_subgraph_embedding(self, ui_states, subg_states, graph):
        # get user, item idx --> vector
        users = graph.ndata['ntype'] == 0
        items = graph.ndata['ntype'] == 1
        subgraphs = graph.ndata['ntype'] == -1

        # print(ui_states[users].shape, ui_states[items].shape, subg_states[subgraphs].shape)

        # x = th.cat([ui_states[users], ui_states[items], subg_states[subgraphs]], 1)
        x = subg_states[subgraphs]
                # if self.subgraph_embedding_from == 'center':
        #     x = states[subgraphs]
        # elif self.subgraph_embedding_from == 'user_item':
        #     x = th.cat([states[users], states[items]], 1)
        # else:
        #     NotImplementedError()

        return x



def edge_drop(graph, edge_dropout=0.2, training=True):
    assert edge_dropout >= 0.0 and edge_dropout <= 1.0, 'Invalid dropout rate.'

    if not training:
        return graph

    # set edge mask to zero in directional mode
    src, _ = graph.edges()
    to_drop = src.new_full((graph.number_of_edges(), ), edge_dropout, dtype=th.float)
    to_drop = th.bernoulli(to_drop).to(th.bool)
    graph.edata['edge_mask'][to_drop] = 0

    return graph