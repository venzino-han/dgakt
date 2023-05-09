import pickle
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler

import config
from data_generator import KT_Sequence_Graph, collate_data


def get_dataloader_junyi(args, data_path="junyi", batch_size=128, num_workers=8, seq_len=64, center_node=True):

    train_df = pd.read_csv(f"data/{data_path}/train_df.csv")
    test_df = pd.read_csv(f"data/{data_path}/test_df.csv")
    problem_df = pd.read_csv(f"data/{data_path}/questions.csv")

    with open(f"data/{data_path}/train_user_group.pkl.zip", 'rb') as pick:
        train_user_group = pickle.load(pick)
    with open(f"data/{data_path}/train_item_group.pkl.zip", 'rb') as pick:
        train_item_group = pickle.load(pick)
    
    train_seq_graph = KT_Sequence_Graph(args, train_user_group, train_item_group, 
                                        interaction_df=train_df,
                                        problem_df=problem_df,
                                        exe_number=config.JUNYI_EXE,
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
                                       exe_number=config.JUNYI_EXE,
                                       seq_len=seq_len,
                                       center_node=center_node,
                                       )
    # test_seq_graph.limit_samples(n=config.JUNYI_TEST_SAMPLES)
    test_loader = DataLoader(test_seq_graph, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             collate_fn=collate_data, pin_memory=True)

    return train_loader, test_loader


def get_dataloader_junyi_part(args, data_path='junyi_part', batch_size=128, num_workers=8, seq_len=64,
                        center_node=True):

    train_df = pd.read_csv(f"data/{data_path}/train_df.csv")
    test_df = pd.read_csv(f"data/{data_path}/test_df.csv")
    problem_df = pd.read_csv(f"data/{data_path}/questions.csv")

    with open(f"data/{data_path}/train_user_group.pkl.zip", 'rb') as pick:
        train_user_group = pickle.load(pick)
    with open(f"data/{data_path}/train_item_group.pkl.zip", 'rb') as pick:
        train_item_group = pickle.load(pick)
    
    train_seq_graph = KT_Sequence_Graph(args, train_user_group, train_item_group, 
                                        interaction_df=train_df,
                                        problem_df=problem_df,
                                        exe_number=config.JUNYI_EXE,
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

    test_seq_graph = KT_Sequence_Graph(args, val_user_group, val_item_group, 
                                       interaction_df=test_df,
                                       problem_df=problem_df,
                                       exe_number=config.JUNYI_EXE,
                                       seq_len=seq_len,
                                       center_node=center_node,
                                       )
    test_seq_graph.filter_part(set([3,4,7]),num=20000)
    test_loader = DataLoader(test_seq_graph, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)

    return train_loader, test_loader


if __name__=="__main__":
    train_loader, _ = get_dataloader(data_path='junyi',
                                     batch_size=32, 
                                     num_workers=8,
                                     seq_len=32
                                     )
    for subg, label in train_loader:
        print(subg)
        print(subg.edata['ts'])
        print(label)
        break

