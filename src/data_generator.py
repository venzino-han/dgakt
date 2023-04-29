# import sys 
# sys.path.append()

import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

from typing import Tuple

import dgl
import torch as th
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import coo_matrix
import config
def one_hot(idx, length):
    x = th.zeros([len(idx), length], dtype=th.float32)
    x[th.arange(len(idx)), idx] = 1.0
    return x

def normalize_timestamp(timestamps, standard_ts):
    timestamps[timestamps == -1.0] = standard_ts
    timestamps = abs(timestamps - standard_ts)
    timestamps = 1 - ((timestamps - th.min(timestamps)) / ((th.max(timestamps) - th.min(timestamps) + 1e-9)))
    return timestamps

def add_center_node(graph, nlabel_vector):
    new_ndata ={
        'ntype': th.tensor([-1], dtype=th.int8),
        'node_id': th.tensor([-1], dtype=th.int32),
    }
    graph.add_nodes(num=1, data=new_ndata)

    ## nodes to connect with center node
    num_new_edges = graph.num_nodes() - 1

    src = [num_new_edges]*num_new_edges + list(range(num_new_edges))
    dst = list(range(num_new_edges)) + [num_new_edges]*num_new_edges

    edata = {
                'etype':th.tensor(np.array([3]*num_new_edges*2), dtype=th.int8),
                'label':th.tensor(np.array([-1]*num_new_edges*2), dtype=th.float32),
                'ts': th.tensor(np.array([-1]*num_new_edges*2), dtype=th.float32),
                'edge_mask': th.tensor(np.array([0]*num_new_edges*2), dtype=th.float32),
                'edge_mask2': th.tensor(np.array([1]*num_new_edges*2), dtype=th.float32),
                'efeat2': th.cat([nlabel_vector, nlabel_vector], dim=0)
            }
            
    graph.add_edges(src, dst, 
        data=edata
    )


"""
subgraph extraction 
"""
def get_subgraph_label(graph:dgl.graph,
                       u_node_idx:th.tensor, i_node_idx:th.tensor,
                       u_neighbors:th.tensor, i_neighbors:th.tensor,
                       part_neighbors:th.tensor,
                       tag_neighbors:th.tensor,
                       center_node:bool = True,
                       )->dgl.graph:
    nodes = th.cat([u_node_idx, i_node_idx, u_neighbors, i_neighbors,
                    part_neighbors, tag_neighbors
                    ], dim=0,) 
    nodes = nodes.type(th.int32)
    nodes = th.clamp(nodes, min=0, max=graph.number_of_nodes()-1)
    subgraph = dgl.node_subgraph(graph, nodes, store_ids=True) 
    node_labels = [0,1] + [2]*len(u_neighbors) + [3]*len(i_neighbors) \
                    + [4]*len(part_neighbors) + [5]*len(tag_neighbors) 
    subgraph.ndata['ntype'] = th.tensor(node_labels, dtype=th.int8)
    nlabel_vector = one_hot(node_labels, 6)
    subgraph.ndata['nlabel'] = one_hot(node_labels, 6)
    subgraph.ndata['x'] = subgraph.ndata['nlabel']

    # set edge mask to zero as to remove links between target nodes in training process
    subgraph = dgl.add_self_loop(subgraph)
    subgraph.edata['edge_mask'] = th.ones(subgraph.number_of_edges(), dtype=th.float32)
    subgraph.edata['edge_mask2'] = th.zeros(subgraph.number_of_edges(), dtype=th.float32)
    target_edges = subgraph.edge_ids([0,1], [1,0], return_uv=False)

    # normalized timestamp
    timestamps = subgraph.edata['ts']
    standard_ts = timestamps[target_edges.to(th.long)[0]].item()
    subgraph.edata['ts'] = normalize_timestamp(timestamps, standard_ts)
    subgraph.remove_edges(target_edges)
    
    ts = subgraph.edata['ts'].unsqueeze(1)
    label = subgraph.edata['label'].unsqueeze(1)
    interaction_counts = subgraph.edata['interaction_counts'].unsqueeze(1)
    subgraph.edata['efeat'] = th.cat([label, ts, interaction_counts], dim=1)

    if center_node :
        # add center node 
        new_ndata ={
            'ntype': th.tensor([-1], dtype=th.int8),
            'node_id': th.tensor([-1], dtype=th.int32),
        }
        subgraph.add_nodes(num=1, data=new_ndata)

        # nodes to connect with center node
        num_new_edges = subgraph.num_nodes() - 1
        src = list(range(num_new_edges))
        dst = [num_new_edges]*num_new_edges

        edata = {
                    'etype':th.tensor(np.array([3]*num_new_edges), dtype=th.int8),
                    'label':th.tensor(np.array([1]*num_new_edges), dtype=th.float32),
                    'ts': th.tensor(np.array([1]*num_new_edges), dtype=th.float32),
                    # 'interaction_counts': th.tensor(np.array([1]*num_new_edges), dtype=th.int32),
                    'edge_mask': th.tensor(np.array([0]*num_new_edges), dtype=th.float32),
                    'edge_mask2': th.tensor(np.array([1]*num_new_edges), dtype=th.float32),
                    'efeat2': nlabel_vector,
                    # 'efeat2': th.cat([nlabel_vector, nlabel_vector], dim=0),
                }
                
        subgraph.add_edges(src, dst, 
            data=edata
        )

    return subgraph


def dedup_edges(srcs, dsts) -> Tuple[list, list]:
    dedup_srcs, dedup_dsts = [], []
    edge_set = set()
    for s, d in zip(srcs, dsts):
        if (s,d) in edge_set:
            continue
        edge_set.add((s,d))
        dedup_srcs.append(s)
        dedup_dsts.append(d)

    return dedup_srcs, dedup_dsts


class KT_Sequence_Graph(Dataset):
    def __init__(self, user_groups, item_groups, interaction_df, problem_df, exe_number=config.EDNET_EXE, seq_len=64, center_node=True):
        self.user_seq_dict = {}
        self.seq_len = seq_len
        self.user_ids = []
        self.user_id_set = set()
        self.center_node = center_node
        self.exe_number = exe_number

        uids = []
        eids = []
        correctness = []
        interaction_count = []

        # get user seqs
        for user_id in tqdm(user_groups.index):
            self.user_id_set.add(user_id)
            exe_id, ans_c, ts, part = user_groups[user_id]
            interaction_c = np.array(list(range(0,len(exe_id))))

            n = len(exe_id)
            uids.extend([user_id]*n)
            eids.extend(exe_id)
            correctness.extend(ans_c)
            interaction_count.extend(interaction_c)

            if len(exe_id) < 2:
                continue

            if len(exe_id) > self.seq_len:
                initial = len(exe_id) % self.seq_len
                if initial > 2:
                    self.user_ids.append(f"{user_id}_0")
                    self.user_seq_dict[f"{user_id}_0"] = (
                        exe_id[:initial], part[:initial], ans_c[:initial], ts[:initial], interaction_c[:initial]
                    )
                chunks = len(exe_id)//self.seq_len
                for c in range(chunks):
                    start = initial + c*self.seq_len
                    end = initial + (c+1)*self.seq_len
                    self.user_ids.append(f"{user_id}_{c+1}")
                    self.user_seq_dict[f"{user_id}_{c+1}"] = (
                        exe_id[start:end], part[start:end], ans_c[start:end], ts[start:end], interaction_c[start:end]
                    )
            else:
                self.user_ids.append(f"{user_id}")
                self.user_seq_dict[f"{user_id}"] = (exe_id, part, ans_c, ts, interaction_c)

        
        print('start item grouping')
        self.item_seq_dict = {}
        
        # def _process_user_seq(user_seq_id):
        #     user_seq = user_seq_dict[user_seq_id]
        #     cids = user_seq[0]
        #     target_cid = cids[-1]
        #     uids = item_groups[target_cid][0]
        #     n = min(self.seq_len, len(uids))
        #     indices = np.random.choice(len(uids), n, replace=False)
        #     return user_seq_id, uids[indices]
        
        # print(datetime.datetime.now())

        # self._process_user_seq_partial = partial(_process_user_seq, self)
        # with Pool(8) as pool:
        #     results = pool.map(self._process_user_seq_partial, self.user_ids)

        # print(datetime.datetime.now())
        # for user_seq_id, uid_seq in results:
        #     self.item_seq_dict[user_seq_id] = uid_seq
        # print(datetime.datetime.now())

        # print(len(self.user_ids))
        for user_seq_id in tqdm(self.user_ids):
            user_seq = self.user_seq_dict[user_seq_id]
            cids = user_seq[0]
            target_cid = cids[-1]
            uids = item_groups[target_cid][0]
            # print(type(uids),uids)
            n = min(self.seq_len, len(uids))
            # indices = np.random.choice(len(uids), n, replace=False)
            self.item_seq_dict[user_seq_id] = uids[-n:]

        # build user-exe graph
        uids = interaction_df['user_id']
        eids = interaction_df['content_id']
        correctness = interaction_df['answered_correctly']
        interaction_counts = interaction_df['interaction_counts']
        ts = interaction_df['timestamp']
        ts = (ts-ts.min())/(ts.max()-ts.min())        

        self.graph = self._build_user_exe_graph(uids, eids, correctness, ts, interaction_counts)

        # exe-KC dict
        num_part = len(set(problem_df['part']))+1
        print('num_part : ', num_part)
        
        self.item_part_dict = {}
        for eid, part in zip(problem_df['question_id'], problem_df['part']):
            self.item_part_dict[eid] = part + self.num_nodes
        parts = []
        for eid in eids:
            parts.append(self.item_part_dict[eid])

        # tag processing
        eids = problem_df['question_id']
        tags = problem_df['tags']
        eid_list = []
        tag_list = []
        for eid, tag_set in zip(eids,tags):
            try:
                tag_set = tag_set.split()
            except:
                continue
            tag_set = map(int, tag_set)
            for tag in tag_set:
                eid_list.append(eid)
                tag_list.append(tag + num_part + self.num_nodes)
        num_tag = len(set(tag_list))
        
        self.item_tag_dict = defaultdict(set)
        for eid, tag in zip(eid_list, tag_list):
            self.item_tag_dict[eid].add(tag)

        print('--------------------------- graph only user-item')
        print(self.graph.number_of_nodes())
        print(self.graph.number_of_edges())
        # eids - parts
        eids_, parts_ = dedup_edges(eids, parts)
        ndata = {
            'ntype': th.tensor(np.array([2]*num_part), dtype=th.int8),
            'node_id': th.tensor(list(range(self.num_nodes, self.num_nodes+num_part)), dtype=th.int32),
        }
        self.graph.add_nodes(num=n, data=ndata)

        parts_ = np.array(parts_)
        src = np.concatenate((eids_, parts_))
        dst = np.concatenate((parts_, eids_))
        edata = {
            'etype':th.tensor(np.array([2]*len(src)), dtype=th.int8),
            'label':th.tensor(np.array([1]*len(src)), dtype=th.float32),
            'ts': th.tensor(np.array([-1]*len(src)), dtype=th.float32),
        }
        self.graph.add_edges(src, dst, 
            data=edata
        )
        print('--------------------------- graph after part')
        print(self.graph.number_of_nodes())
        print(self.graph.number_of_edges())

        eids_, tags_ = dedup_edges(eids, tag_list)
        ndata = {
            'ntype': th.tensor(np.array([3]*num_tag), dtype=th.int8),
            'node_id': th.tensor(list(range(self.num_nodes+num_part, self.num_nodes+num_part+num_tag)), dtype=th.int32),
        }
        self.graph.add_nodes(num=n, data=ndata)
        
        tags_ = np.array(tags_)
        src = np.concatenate((eids_, tags_))
        dst = np.concatenate((tags_, eids_))
        edata = {
            'etype':th.tensor(np.array([3]*len(src)), dtype=th.int8),
            'label':th.tensor(np.array([1]*len(src)), dtype=th.float32),
            'ts': th.tensor(np.array([-1]*len(src)), dtype=th.float32),
        }
        self.graph.add_edges(src, dst, 
            data=edata
        )
        print('--------------------------- graph after tag')
        print(self.graph.number_of_nodes())
        print(self.graph.number_of_edges())

    def filter_part(self, part_set=set()):
        print('user_ids :', len(self.user_ids))
        
        new_user_ids_list = []
        for user_seq_id in self.user_ids:
            exe_id, part, ans_c, ts, interaction_counts = self.user_seq_dict[user_seq_id]
            target_item_part = part[-1]
            if target_item_part in part_set:
                new_user_ids_list.append(user_seq_id)
            else:
                del self.user_seq_dict[user_seq_id]

        self.user_ids = new_user_ids_list
        print('filterd user_ids :', len(self.user_ids))

    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, index):
        user_seq_id = self.user_ids[index]
        exe_id, part, ans_c, ts, interaction_counts = self.user_seq_dict[user_seq_id]
        u_id = self.item_seq_dict[user_seq_id]

        #build graph
        label = ans_c[1:]
        label = np.clip(label, 0, 1)
        
        target_item_id = exe_id[-1]
        label = label[-1]


        #build graph
        u_idx, i_idx = int(user_seq_id.split('_')[0])+self.exe_number, target_item_id
        u_neighbors, i_neighbors = u_id+self.exe_number, exe_id[:-1]
        u_neighbors = u_neighbors[u_neighbors!=i_idx]
        i_neighbors = i_neighbors[i_neighbors!=u_idx]

        part_neighbors = [self.item_part_dict[i_idx]] + [self.item_part_dict[i_neighbor] for i_neighbor in i_neighbors]
        # tag_neighbors = [self.item_tag_dict[i_idx]] + [self.item_tag_dict[i_neighbor] for i_neighbor in i_neighbors]
        tag_neighbors = list(self.item_tag_dict[i_idx])

        for i_neighbor in i_neighbors:
            tag_neighbors += list(self.item_tag_dict[i_neighbor])

        part_neighbors = list(set(part_neighbors))
        tag_neighbors = list(set(tag_neighbors))

        subgraph = get_subgraph_label(graph = self.graph,
                                      u_node_idx=th.tensor([u_idx]),
                                      i_node_idx=th.tensor([i_idx]),
                                      u_neighbors=th.tensor(u_neighbors),
                                      i_neighbors=th.tensor(i_neighbors),
                                      part_neighbors=th.tensor(part_neighbors),
                                      tag_neighbors=th.tensor(tag_neighbors),
                                      center_node=self.center_node,
                                    )

        n = subgraph.number_of_edges()
        if n > 4000:
            prob = (n-4000)/n
            transform = dgl.DropEdge(prob)
            subgraph = transform(subgraph)


        return subgraph, th.tensor(label, dtype=th.float32)
    
    def _build_user_exe_graph(self, uids, eids, correctness, ts, interaction_counts):
        self.num_user = max(uids)+1
        uids += self.exe_number
        self.num_nodes = self.num_user+self.exe_number

        interaction_counts = np.clip(interaction_counts, a_min=0, a_max=128)/128

        src_nodes = np.concatenate((uids, eids))
        dst_nodes = np.concatenate((eids, uids))
        correctness = np.concatenate((correctness, correctness))
        interaction_counts = np.concatenate((interaction_counts, interaction_counts))
        ts = np.concatenate((ts, ts))
        # print(len(src_nodes), len(dst_nodes), len(correctness))

        user_exe_matrix = coo_matrix((correctness, (src_nodes, dst_nodes)), shape=(self.num_nodes, self.num_nodes))
        
        # build graph 
        graph = dgl.from_scipy(sp_mat=user_exe_matrix, idtype=th.int32)
        graph.ndata['node_id'] = th.tensor(list(range(self.num_nodes)), dtype=th.int32)
        graph.ndata['ntype'] = th.tensor(self.exe_number*[0]+self.num_user*[1], dtype=th.int8)
        
        graph.edata['label'] = th.tensor(correctness, dtype=th.float32)
        graph.edata['etype'] = th.tensor(correctness, dtype=th.int8)
        graph.edata['ts'] = th.tensor(ts, dtype=th.float32)
        graph.edata['interaction_counts'] = th.tensor(interaction_counts, dtype=th.float32)

        return graph

def collate_data(data):
    g_list, label_list = map(list, zip(*data))
    g = dgl.batch(g_list)
    g_label = th.stack(label_list)
    return g, g_label


