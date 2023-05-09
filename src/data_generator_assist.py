# import sys 
# sys.path.append()

import pickle
import pandas as pd

from torch.utils.data import DataLoader

import config
from data_generator import KT_Sequence_Graph, collate_data


def get_dataloader_assist(args, data_path='assist', batch_size=128, 
                          num_workers=8, seq_len=64, center_node=True):

    train_df = pd.read_csv(f'data/{data_path}/train_df.csv')
    test_df = pd.read_csv(f'data/{data_path}/test_df.csv')
    problem_df = pd.read_csv(f'data/{data_path}/questions.csv')

    with open(f"data/{data_path}/train_user_group.pkl.zip", 'rb') as pick:
        train_user_group = pickle.load(pick)
    with open(f"data/{data_path}/train_item_group.pkl.zip", 'rb') as pick:
        train_item_group = pickle.load(pick)
    
    train_seq_graph = KT_Sequence_Graph(args, train_user_group, train_item_group, 
                                        interaction_df=train_df,
                                        problem_df=problem_df,
                                        exe_number=config.ASSIST_EXE,
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

    test_seq_graph = KT_Sequence_Graph(args, val_user_group, val_item_group,
                                       interaction_df=test_df,
                                       problem_df=problem_df,
                                       exe_number=config.ASSIST_EXE,
                                       seq_len=seq_len,
                                       center_node=center_node,
                                       )
    test_loader = DataLoader(test_seq_graph, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)

    return train_loader, test_loader


def get_dataloader_assist_part(args, data_path='assist_part', batch_size=128, 
                                num_workers=8, seq_len=64, center_node=True):

    train_df = pd.read_csv(f'data/{data_path}/train_df.csv')
    test_df = pd.read_csv(f'data/{data_path}/test_df.csv')
    problem_df = pd.read_csv(f'data/{data_path}/questions.csv')

    with open(f"data/{data_path}/train_user_group.pkl.zip", 'rb') as pick:
        train_user_group = pickle.load(pick)
    with open(f"data/{data_path}/train_item_group.pkl.zip", 'rb') as pick:
        train_item_group = pickle.load(pick)
    
    train_seq_graph = KT_Sequence_Graph(args, train_user_group, train_item_group, 
                                        interaction_df=train_df,
                                        problem_df=problem_df,
                                        exe_number=config.ASSIST_EXE,
                                        seq_len=seq_len,
                                        center_node=center_node,
                                        )
    train_seq_graph.filter_part(set([0,1,2,3,4,5,6]))
    train_loader = DataLoader(train_seq_graph, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)

    print('start validation graph generation')

    with open(f"data/{data_path}/val_user_group.pkl.zip", 'rb') as pick:
        val_user_group = pickle.load(pick)
    with open(f"data/{data_path}/val_item_group.pkl.zip", 'rb') as pick:
        val_item_group = pickle.load(pick)

    test_seq_graph = KT_Sequence_Graph(args, val_user_group, val_item_group,
                                       interaction_df=test_df,
                                       problem_df=problem_df,
                                       exe_number=config.ASSIST_EXE,
                                       seq_len=seq_len,
                                       center_node=center_node,
                                       )
    test_seq_graph.filter_part(set([7,8,9,10]))
    test_loader = DataLoader(test_seq_graph, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)

    return train_loader, test_loader



if __name__=="__main__":
    """
    Data loader test
    """
    from tqdm import tqdm
    from easydict import EasyDict
    args = EasyDict({
            'use_ts': True,
            'use_count': True,
    })

    # train_loader, test_loader = get_dataloader_assist(args, batch_size=1, seq_len=64, num_workers=1)

    # for i, (batch, label) in enumerate(test_loader):
    #     print(i)
    # #     # print(batch.ndata['ntype'])
    # #     # print(batch.edata['etype'])
    # #     # print(batch.edata['label'])
    # #     # print(batch.edata['ts'])
    # #     # print(batch.ndata['x'])
    # #     # print(label.shape)
    # #     # print(label)
    # #     # break
    #     pass
    
    data_path='assist'
    seq_len=64
    center_node=True

    train_df = pd.read_csv(f'data/{data_path}/train_df.csv')
    test_df = pd.read_csv(f'data/{data_path}/test_df.csv')
    problem_df = pd.read_csv(f'data/{data_path}/questions.csv')

    # with open(f"data/{data_path}/train_user_group.pkl.zip", 'rb') as pick:
    #     train_user_group = pickle.load(pick)
    # with open(f"data/{data_path}/train_item_group.pkl.zip", 'rb') as pick:
    #     train_item_group = pickle.load(pick)
    
    # train_seq_graph = KT_Sequence_Graph(args, train_user_group, train_item_group, 
    #                                     interaction_df=train_df,
    #                                     problem_df=problem_df,
    #                                     exe_number=config.ASSIST_EXE,
    #                                     seq_len=seq_len,
    #                                     center_node=center_node,
    #                                     )
    # train_loader = DataLoader(train_seq_graph, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    #                             collate_fn=collate_data, pin_memory=True)

    print('start validation graph generation')

    with open(f"data/{data_path}/val_user_group.pkl.zip", 'rb') as pick:
        val_user_group = pickle.load(pick)
    with open(f"data/{data_path}/val_item_group.pkl.zip", 'rb') as pick:
        val_item_group = pickle.load(pick)

    test_seq_graph = KT_Sequence_Graph(args, val_user_group, val_item_group,
                                       interaction_df=test_df,
                                       problem_df=problem_df,
                                       exe_number=config.ASSIST_EXE,
                                       seq_len=seq_len,
                                       center_node=center_node,
                                       )
    # test_loader = DataLoader(test_seq_graph, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    #                             collate_fn=collate_data, pin_memory=True)
    # print('len : ', len(test_seq_graph))
    # print('uid : ',test_seq_graph.user_ids[3343])
    # print(test_seq_graph[3343])
    print('uid : ',test_seq_graph.user_ids[3344])
    print(test_seq_graph[3344])