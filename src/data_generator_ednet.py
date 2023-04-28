import pickle
import pandas as pd
from torch.utils.data import DataLoader

import config
from data_generator import KT_Sequence_Graph, collate_data

# class KT_Sequence_Graph(Dataset):
#     def __init__(self, user_groups, item_groups, interaction_df, problem_df, seq_len, center_node):
#         self.user_seq_dict = {}
#         self.seq_len = seq_len
#         self.user_ids = []
#         self.user_id_set = set()
#         self.center_node = center_node

#         uids = []
#         eids = []
#         correctness = []

#         # get user seqs
#         for user_id in user_groups.index:
#             self.user_id_set.add(user_id)
#             c_id, part, t_c_id, t_lag, q_et, ans_c, q_he, u_ans, ts = user_groups[user_id]

#             n = len(c_id)
#             uids.extend([user_id]*n)
#             eids.extend(c_id)
#             correctness.extend(ans_c)

#             if len(c_id) < 2:
#                 continue

#             if len(c_id) > self.seq_len:
#                 initial = len(c_id) % self.seq_len
#                 if initial > 2:
#                     self.user_ids.append(f"{user_id}_0")
#                     self.user_seq_dict[f"{user_id}_0"] = (
#                         c_id[:initial], part[:initial], ans_c[:initial], ts[:initial]
#                     )
#                 chunks = len(c_id)//self.seq_len
#                 for c in range(chunks):
#                     start = initial + c*self.seq_len
#                     end = initial + (c+1)*self.seq_len
#                     self.user_ids.append(f"{user_id}_{c+1}")
#                     self.user_seq_dict[f"{user_id}_{c+1}"] = (
#                         c_id[start:end], part[start:end], ans_c[start:end], ts[start:end]
#                     )
#             else:
#                 self.user_ids.append(f"{user_id}")
#                 self.user_seq_dict[f"{user_id}"] = (c_id, part, ans_c, ts)
        
#         self.item_seq_dict = {}
#         for user_seq_id in self.user_ids:
#             user_seq = self.user_seq_dict[user_seq_id]
#             target_cid = user_seq[0]
#             target_cid = target_cid[-1]
#             u_id, part, t_c_id, t_lag, q_et, ans_c, q_he, u_ans, ts = item_groups[target_cid]
#             n = self.seq_len #*2
#             if n > len(u_id):
#                 n =len(u_id)
#             indices = np.random.choice(len(u_id), n, replace=False)
#             self.item_seq_dict[user_seq_id] = u_id[indices]


#         # build user-exe graph
#         uids = interaction_df['user_id']
#         eids = interaction_df['content_id'] - 1
#         correctness = interaction_df['answered_correctly'] - 1
#         ts = interaction_df['timestamp']
#         ts = (ts-ts.min())/(ts.max()-ts.min())

#         self.graph = self._build_user_exe_graph(uids, eids, correctness, ts)

#         # exe-KC dict
#         num_part = max(problem_df['part']) + 1
        
#         self.item_part_dict = {}
#         for eid, part in zip(problem_df['question_id'], problem_df['part']):
#             self.item_part_dict[eid] = part + self.num_nodes
#         parts = []
#         for eid in eids:
#             parts.append(self.item_part_dict[eid])

#         # tag processing
#         pids = problem_df['question_id']
#         tags = problem_df['tags']
#         eid_list = []
#         tag_list = []
#         for pid, tag_set in zip(pids,tags):
#             try:
#                 tag_set = tag_set.split()
#             except:
#                 continue
#             tag_set = map(int, tag_set)
#             for tag in tag_set:
#                 eid_list.append(pid)
#                 tag_list.append(tag + num_part + self.num_nodes)
#         num_tag = len(set(tag_list))
        
#         self.item_tag_dict = defaultdict(set)
#         for eid, tag in zip(eid_list, tag_list):
#             self.item_tag_dict[eid].add(tag)


#         print('--------------------------- graph only user-item')
#         print(self.graph.number_of_nodes())
#         print(self.graph.number_of_edges())
#         # eids - parts
#         eids_, parts_ = dedup_edges(eids, parts)
#         ndata = {
#             'ntype': th.tensor(np.array([2]*num_part), dtype=th.int8),
#             'node_id': th.tensor(list(range(self.num_nodes, self.num_nodes+num_part)), dtype=th.int32),
#         }
#         self.graph.add_nodes(num=n, data=ndata)

#         parts_ = np.array(parts_)
#         src = np.concatenate((eids_, parts_))
#         dst = np.concatenate((parts_, eids_))
#         edata = {
#             'etype':th.tensor(np.array([2]*len(src)), dtype=th.int8),
#             'label':th.tensor(np.array([1]*len(src)), dtype=th.float32),
#             'ts': th.tensor(np.array([-1]*len(src)), dtype=th.float32),
#         }
#         self.graph.add_edges(src, dst, 
#             data=edata
#         )
#         print('--------------------------- graph after part')
#         print(self.graph.number_of_nodes())
#         print(self.graph.number_of_edges())

#         eids_, tags_ = dedup_edges(eids, tag_list)
#         ndata = {
#             'ntype': th.tensor(np.array([3]*num_tag), dtype=th.int8),
#             'node_id': th.tensor(list(range(self.num_nodes+num_part, self.num_nodes+num_part+num_tag)), dtype=th.int32),
#         }
#         self.graph.add_nodes(num=n, data=ndata)
        
#         tags_ = np.array(tags_)
#         src = np.concatenate((eids_, tags_))
#         dst = np.concatenate((tags_, eids_))
#         edata = {
#             'etype':th.tensor(np.array([3]*len(src)), dtype=th.int8),
#             'label':th.tensor(np.array([1]*len(src)), dtype=th.float32),
#             'ts': th.tensor(np.array([-1]*len(src)), dtype=th.float32),
#         }
#         self.graph.add_edges(src, dst, 
#             data=edata
#         )
#         print('--------------------------- graph after tag')
#         print(self.graph.number_of_nodes())
#         print(self.graph.number_of_edges())

#     def filter_part(self, part_set=set(), num=50000):
#         self.user_ids = self.user_ids[:num]
#         print('user_ids :', len(self.user_ids))

#         parts = set()
        
#         new_user_ids_list = []
#         for user_seq_id in self.user_ids:
#             c_id, part, ans_c, ts = self.user_seq_dict[user_seq_id]
#             target_item_part = part[-1]
#             parts.add(target_item_part)
#             if target_item_part in part_set:
#                 new_user_ids_list.append(user_seq_id)
#             else:
#                 del self.user_seq_dict[user_seq_id]

#         self.user_ids = new_user_ids_list
#         print(parts)
#         print('filterd user_ids :', len(self.user_ids))


#     def __len__(self):
#         return len(self.user_ids)
    
#     def __getitem__(self, index):
#         user_seq_id = self.user_ids[index]
#         c_id, part, ans_c, ts = self.user_seq_dict[user_seq_id]
#         u_id = self.item_seq_dict[user_seq_id]

#         #build graph
#         label = ans_c[1:] - 1
#         label = np.clip(label, 0, 1)
        
#         target_item_id = c_id[-1]
#         label = label[-1]

#         #build graph
#         u_idx, i_idx = int(user_seq_id.split('_')[0])+config.TOTAL_EXE, target_item_id-1
#         u_neighbors, i_neighbors = u_id+config.TOTAL_EXE, c_id[:-1]-1
#         u_neighbors = u_neighbors[u_neighbors!=i_idx]
#         i_neighbors = i_neighbors[i_neighbors!=u_idx]

#         part_neighbors = [self.item_part_dict[i_idx]] + [self.item_part_dict[i_neighbor] for i_neighbor in i_neighbors]
#         tag_neighbors = list(self.item_tag_dict[i_idx])

#         for i_neighbor in i_neighbors:
#             tag_neighbors += list(self.item_tag_dict[i_neighbor])

#         part_neighbors = list(set(part_neighbors))
#         tag_neighbors = list(set(tag_neighbors))

#         subgraph = get_subgraph_label(graph = self.graph,
#                                       u_node_idx=th.tensor([u_idx]),
#                                       i_node_idx=th.tensor([i_idx]),
#                                       u_neighbors=th.tensor(u_neighbors),
#                                       i_neighbors=th.tensor(i_neighbors),
#                                       part_neighbors=th.tensor(part_neighbors),
#                                       tag_neighbors=th.tensor(tag_neighbors),
#                                       center_node=self.center_node,
#                                     )
        

#         return subgraph, th.tensor(label, dtype=th.float32)
    
#     def _build_user_exe_graph(self, uids, eids, correctness, ts):
#         self.num_user = max(uids)+1
#         uids += config.TOTAL_EXE
#         self.num_nodes = self.num_user+config.TOTAL_EXE

#         src_nodes = np.concatenate((uids, eids))
#         dst_nodes = np.concatenate((eids, uids))
#         correctness = np.concatenate((correctness, correctness))
#         ts = np.concatenate((ts, ts))
#         print(len(src_nodes), len(dst_nodes), len(correctness))
#         user_exe_matrix = coo_matrix((correctness, (src_nodes, dst_nodes)), shape=(self.num_nodes, self.num_nodes))
        
#         # build graph 
#         graph = dgl.from_scipy(sp_mat=user_exe_matrix, idtype=th.int32)
#         graph.ndata['node_id'] = th.tensor(list(range(self.num_nodes)), dtype=th.int32)
#         graph.ndata['ntype'] = th.tensor(config.TOTAL_EXE*[0]+self.num_user*[1], dtype=th.int8)
        
#         graph.edata['label'] = th.tensor(correctness, dtype=th.float32)
#         graph.edata['etype'] = th.tensor(correctness, dtype=th.int8)
#         graph.edata['ts'] = th.tensor(ts, dtype=th.float32)

#         return graph

def get_dataloader_ednet(data_path="ednet", batch_size=128, num_workers=8, seq_len=64,center_node=True):

    train_df = pd.read_csv(f"data/{data_path}/train_df.csv")
    test_df = pd.read_csv(f"data/{data_path}/test_df.csv")
    problem_df = pd.read_csv(f"data/{data_path}/questions.csv")

    with open(f"data/{data_path}/train_user_group.pkl.zip", 'rb') as pick:
        train_user_group = pickle.load(pick)
    with open(f"data/{data_path}/train_item_group.pkl.zip", 'rb') as pick:
        train_item_group = pickle.load(pick)
    
    train_seq_graph = KT_Sequence_Graph(train_user_group, train_item_group, 
                                        interaction_df=train_df,
                                        problem_df=problem_df,
                                        exe_number=config.EDNET_EXE,
                                        seq_len=seq_len,
                                        center_node=center_node, 
                                        )
    train_loader = DataLoader(train_seq_graph, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)

    print('start validation graph generation')

    with open(f"data/{data_path}/val_user_group.pkl.zip", 'rb') as pick:
        val_user_group = pickle.load(pick)
    with open(f"data/{data_path}/val_item_group.pkl.zip", 'rb') as pick:
        val_item_group = pickle.load(pick)

    test_seq_graph = KT_Sequence_Graph(val_user_group, val_item_group, 
                                       interaction_df=test_df,
                                       problem_df=problem_df,
                                       exe_number=config.EDNET_EXE,
                                       seq_len=seq_len,
                                       center_node=center_node,
                                       )

    test_loader = DataLoader(test_seq_graph, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)

    return train_loader, test_loader


def get_dataloader_ednet_part(data_path='ednet_part', batch_size=128, num_workers=8, seq_len=64,
                        center_node=True):

    train_df = pd.read_csv(f"data/{data_path}/train_df.csv")
    test_df = pd.read_csv(f"data/{data_path}/test_df.csv")
    problem_df = pd.read_csv(f"data/{data_path}/questions.csv")

    with open(f"data/{data_path}/train_user_group.pkl.zip", 'rb') as pick:
        train_user_group = pickle.load(pick)
    with open(f"data/{data_path}/train_item_group.pkl.zip", 'rb') as pick:
        train_item_group = pickle.load(pick)
    
    train_seq_graph = KT_Sequence_Graph(train_user_group, train_item_group, 
                                        interaction_df=train_df,
                                        problem_df=problem_df,
                                        exe_number=config.EDNET_EXE,
                                        seq_len=seq_len,
                                        center_node=center_node,
                                        )
    train_seq_graph.filter_part(set([1,2,5,6]), num=80000)
    train_loader = DataLoader(train_seq_graph, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)

    print('start validation graph generation')

    with open(f"data/{data_path}/val_user_group.pkl.zip", 'rb') as pick:
        val_user_group = pickle.load(pick)
    with open(f"data/{data_path}/val_item_group.pkl.zip", 'rb') as pick:
        val_item_group = pickle.load(pick)

    test_seq_graph = KT_Sequence_Graph(val_user_group, val_item_group, 
                                       interaction_df=test_df,
                                       problem_df=problem_df,
                                       exe_number=config.EDNET_EXE,
                                       seq_len=seq_len,
                                       center_node=center_node,
                                       )
    test_seq_graph.filter_part(set([3,4,7]),num=20000)
    test_loader = DataLoader(test_seq_graph, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)

    return train_loader, test_loader


if __name__=="__main__":
    train_loader, _ = get_dataloader(data_path='ednet',
                                     batch_size=32, 
                                     num_workers=8,
                                     seq_len=32
                                     )
    for subg, label in train_loader:
        print(subg)
        print(subg.edata['ts'])
        print(label)
        break

