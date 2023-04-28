"""Dual Attention Graph Knowledge Tracing modules v2"""
import torch as th
import torch.nn as nn
from .egatconv import EGATConv

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
        self.agg_layers_1 = th.nn.ModuleList()
        self.global_egats = th.nn.ModuleList()
        self.agg_layers_2 = th.nn.ModuleList()
        latent_dim = copy.deepcopy(latent_dim)
        latent_dim.insert(0, in_nfeats)

        for i in range(0, len(latent_dim)-1):
            self.local_egats.append(EGATConv(in_node_feats=latent_dim[i],
                                            in_edge_feats=in_efeats,
                                            out_node_feats=latent_dim[i+1],
                                            out_edge_feats=16,
                                            # out_edge_feats=latent_dim[i+1],
                                            num_heads=num_heads))
            self.agg_layers_1.append(nn.Linear(latent_dim[i+1]*num_heads, latent_dim[i+1]))
            self.global_egats.append(EGATConv(in_node_feats=latent_dim[i+1],
                                            in_edge_feats=in_nfeats,
                                            out_node_feats=latent_dim[i+1],
                                            out_edge_feats=16,
                                            # out_edge_feats=latent_dim[i+1],
                                            num_heads=num_heads))
            self.agg_layers_2.append(nn.Linear(latent_dim[i+1]*num_heads, latent_dim[i+1]))
            

        self.lin1_ui = nn.Linear(sum(latent_dim[1:])*2, 128)
        self.lin1_subg = nn.Linear(sum(latent_dim[1:])*1, 128)
        self.dropout = nn.Dropout(0.3)
        self.lin2_ui = nn.Linear(128, 1)
        self.lin2_subg = nn.Linear(128, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1_ui.reset_parameters()
        self.lin1_subg.reset_parameters()
        self.lin2_ui.reset_parameters()
        self.lin2_subg.reset_parameters()
        for l1, l2 in zip(self.agg_layers_1, self.agg_layers_2):
            l1.reset_parameters()
            l2.reset_parameters()

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
        e = graph.edata['efeat']
        mask = graph.edata['edge_mask']
        mask2 = graph.edata['edge_mask2']
        e2 = graph.edata['efeat2']
        subg_states = []
        ui_states = []

        for local_egat, global_egat, agg_lin1, agg_lin2 in zip(self.local_egats, self.global_egats, self.agg_layers_1, self.agg_layers_2):
            x, _ = local_egat(graph=graph, nfeats=x, efeats=e, norm=mask)
            x = x.flatten(1, 2) 
            x = agg_lin1(x)
            x = self.elu(x)
            ui_states.append(x)

            x, _, attention = global_egat(graph=graph, nfeats=x, efeats=e2, norm=mask2, get_attention= True)
            self.attention = attention
            x = x.flatten(1, 2) 
            x = agg_lin2(x)
            # x = x.sum(dim=1)
            x = self.leakyrelu(x)
            graph.ndata['x'] = x
            subg_states.append(x)

        ui_states = th.cat(ui_states, 1)
        subg_states = th.cat(subg_states, 1)
        ui_x, subg_x = self._get_subgraph_embedding(ui_states, subg_states, graph)
        
        subg_x = th.relu(self.lin1_subg(subg_x))
        ui_x = th.relu(self.lin1_ui(ui_x))
        
        subg_x = self.dropout(subg_x)
        ui_x = self.dropout(ui_x)
        
        subg_x = self.lin2_subg(subg_x)
        ui_x = self.lin2_ui(ui_x)
        
        # x = th.sigmoid(x)
        subg_x = th.sigmoid(subg_x)
        ui_x = th.sigmoid(ui_x)

        return subg_x[:, 0], ui_x[:, 0]

    def _get_subgraph_embedding(self, ui_states, subg_states, graph):
        # get user, item idx --> vector
        users = graph.ndata['ntype'] == 0
        items = graph.ndata['ntype'] == 1
        subgraphs = graph.ndata['ntype'] == -1

        # print(ui_states[users].shape, ui_states[items].shape, subg_states[subgraphs].shape)

        ui_x = th.cat([ui_states[users], ui_states[items]], 1)
        subg_x = subg_states[subgraphs]

                # if self.subgraph_embedding_from == 'center':
        #     x = states[subgraphs]
        # elif self.subgraph_embedding_from == 'user_item':
        #     x = th.cat([states[users], states[items]], 1)
        # else:
        #     NotImplementedError()

        return ui_x, subg_x



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