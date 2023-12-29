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
    try:
        val_df = pd.read_csv(f'data/{data_path}/val_df.csv')
    except:
        val_df = None
        val_loader = None
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

    if val_df is not None:
        print('start validation graph generation')

        with open(f"data/{data_path}/val_user_group.pkl.zip", 'rb') as pick:
            val_user_group = pickle.load(pick)
        with open(f"data/{data_path}/val_item_group.pkl.zip", 'rb') as pick:
            val_item_group = pickle.load(pick)

        val_seq_graph = KT_Sequence_Graph(args, val_user_group, val_item_group,
                                        interaction_df=val_df,
                                        problem_df=problem_df,
                                        exe_number=config.ASSIST_EXE,
                                        seq_len=seq_len,
                                        center_node=center_node,
                                        )
        val_loader = DataLoader(val_seq_graph, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                    collate_fn=collate_data, pin_memory=True)

    print('start test graph generation')

    with open(f"data/{data_path}/test_user_group.pkl.zip", 'rb') as pick:
        test_user_group = pickle.load(pick)
    with open(f"data/{data_path}/test_item_group.pkl.zip", 'rb') as pick:
        test_item_group = pickle.load(pick)

    test_seq_graph = KT_Sequence_Graph(args, test_user_group, test_item_group,
                                       interaction_df=test_df,
                                       problem_df=problem_df,
                                       exe_number=config.ASSIST_EXE,
                                       seq_len=seq_len,
                                       center_node=center_node,
                                       )
    test_loader = DataLoader(test_seq_graph, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)

    return train_loader, val_loader, test_loader


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

