"""Torch modules for graph attention networks v2 (GATv2)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn.functional import edge_softmax

class GraphLevelAttention(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 bias=True,
                 ):
        super(GraphLevelAttention, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = in_feats, in_feats 
        self._out_feats = out_feats

        self.fc_src = nn.Linear(
            self._in_src_feats, out_feats * num_heads, bias=bias)

        self.fc_dst = nn.Linear(
            self._in_src_feats, out_feats * num_heads, bias=bias)

        self.attn = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.register_buffer('res_fc', None)
        self.activation = activation
        self.bias = bias
        self.reset_parameters()


    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        if self.bias:
            nn.init.constant_(self.fc_src.bias, 0)
            nn.init.constant_(self.fc_dst.bias, 0)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn, gain=gain)


    def message(self, edges):
        """Message function."""
        m = self.linear_r(edges.src['h'], edges.data['etype'], self.presorted)
        if 'norm' in edges.data:
            m = m * edges.data['norm']
        return {'m' : m}


    def forward(self, graph, feat, norm=None, get_attention=False):

        with graph.local_scope():

            h_src = h_dst = self.feat_drop(feat)
            feat_src = self.fc_src(h_src).view(
                -1, self._num_heads, self._out_feats)

            feat_dst = self.fc_dst(h_src).view(
                -1, self._num_heads, self._out_feats)
            
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]
                h_dst = h_dst[:graph.number_of_dst_nodes()]

            graph.srcdata.update({'el': feat_src})# (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': feat_dst})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))# (num_src_edge, num_heads, out_dim)
            
            if norm is not None:
                # print(e.shape, norm.tile(self._out_feats, self._num_heads,1).T.shape)
                e = e * norm.tile(self._out_feats, self._num_heads, 1).T

            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)# (num_edge, num_heads, 1)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)) # (num_edge, num_heads)
            # message passing
            ## TODO : update node feature with attention 
            graph.update_all(fn.u_mul_e('el', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
