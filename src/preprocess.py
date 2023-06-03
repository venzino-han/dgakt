"""
pre-processing dataset
"""

import time
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from copy import copy

def group_seq(df, groupby_key, cols, save_path):
    # build group : user_id(item_id) - seq
    # save as file
    cols = copy(cols)
    cols.remove(groupby_key)
    print(cols)
    group = df.groupby(groupby_key).apply(lambda df: tuple([df[c].values for c in cols]))
    with open(save_path, 'wb') as pick:
        pickle.dump(group, pick)
    del group, df
    return

def save_user_item_group(data_path, train_features=["user_id", "content_id", "answered_correctly", "timestamp", 'part']):
    print("Start group pre-process")
    t_s = time.time()
    train_path = f'data/{data_path}/train_df.csv'
    test_path = f'data/{data_path}/test_df.csv'
    train_df = pd.read_csv(train_path)[train_features]
    test_df = pd.read_csv(test_path)[train_features]
    print(len(train_df), len(test_df))
    

    group_seq(df=train_df, groupby_key="user_id", cols=train_features, save_path=f"data/{data_path}/train_user_group.pkl.zip")
    group_seq(df=train_df, groupby_key="content_id", cols=train_features, save_path=f"data/{data_path}/train_item_group.pkl.zip")

    group_seq(df=test_df, groupby_key="user_id", cols=train_features, save_path=f"data/{data_path}/val_user_group.pkl.zip")
    group_seq(df=test_df, groupby_key="content_id", cols=train_features, save_path=f"data/{data_path}/val_item_group.pkl.zip")
    
    print("Complete grouping, execution time {:.2f} s".format(time.time()-t_s))


def count_interactions(df):
    user_count_dict = defaultdict(int)
    interaction_counts = np.zeros(len(df))
    for i, uid in enumerate(df.user_id):
        temp_count = user_count_dict.get(uid,0)
        interaction_counts[i] = temp_count
        user_count_dict[uid]+=1
    return interaction_counts

def pre_process_df(train_path, ques_path, split_ratio=0.8):
    print("Start pre-process")
    t_s = time.time()
    df = pd.read_csv(train_path)
    df = df.query('answered_correctly!=-1')
    df = df.sort_values(by=["timestamp"])

    # add interaction count for each users
    interaciton_counts = count_interactions(df)
    df['interaction_counts'] = interaciton_counts
    
    # merge with question dataframe to get part feature
    print("Start merge question dataframe")
    ques_df = pd.read_csv(ques_path)[["question_id", "part", "tags"]]
    df = df.merge(ques_df, how='left', left_on='content_id', right_on='question_id')
    df.drop(["question_id"], axis=1, inplace=True)

    print("Complete merge dataframe")
    print("====================")
    print(df.head(10))

    num_rows = df.shape[0]
    val_df = df[int(num_rows*split_ratio):]
    train_df = df[:int(num_rows*split_ratio)]

    print("Train dataframe shape after process ({}, {})/ Val dataframe shape after process({}, {})".format(train_df.shape[0], train_df.shape[1], val_df.shape[0], val_df.shape[1]))
    print("====================")

    """check data balance"""
    num_new_user = val_df[~val_df["user_id"].isin(train_df["user_id"])]["user_id"].nunique()
    num_new_content = val_df[~val_df["content_id"].isin(train_df["content_id"])]["content_id"].nunique()
    train_content_id = train_df["content_id"].nunique()
    train_part = train_df["part"].nunique()
    train_correct = train_df["answered_correctly"].mean()
    val_correct = val_df["answered_correctly"].mean()
    print("Number of new users {}/ Number of new contents {}".format(num_new_user, num_new_content))
    print("Number of content_id {}/ Number of part {}".format(train_content_id, train_part))
    print("train correctness {:.3f}/val correctness {:.3f}".format(train_correct, val_correct))
    print("====================")
    print("Complete pre-process, execution time {:.2f} s".format(time.time()-t_s))

    return train_df, df



if __name__=="__main__":

    # EDNET_TRAIN_FEATURES = ["user_id", "content_id", "part", "task_container_id", "time_lag", "prior_question_elapsed_time",
    #                         "answered_correctly", "prior_question_had_explanation", "user_answer", "timestamp"]
    BASIC_TRAIN_FEATURES = ["user_id", "content_id", "answered_correctly", "timestamp", 'part']
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset", action="store", default="ednet")
    parser.add_argument("-s", "--savepath", dest="filepath", action="store", default="ednet")
    args = parser.parse_args()
    
    assert args.dataset in {'ednet', 'assist','junyi'
                            # 'ednet_part','assist_part',
                            }


    train_path_dict = {
        'ednet':'data/ednet/train_30m.csv',
        'assist':'data/assist/assist.csv',
        'junyi':'data/junyi/junyi.csv'
    }

    question_path_dict = {
        'ednet':'data/ednet/questions.csv',
        'assist':'data/assist/questions.csv',
        'junyi':'data/junyi/questions.csv'
    }


    train_path = train_path_dict.get(args.dataset)
    question_path = question_path_dict.get(args.dataset)

    
    train_df, test_df = pre_process_df(train_path, question_path)
    train_df, test_df = train_df[BASIC_TRAIN_FEATURES+['interaction_counts']], test_df[BASIC_TRAIN_FEATURES+['interaction_counts']]
    train_df.to_csv(f'data/{args.dataset}/train_df.csv')
    test_df.to_csv(f'data/{args.dataset}/test_df.csv')
    
    save_user_item_group(data_path=args.dataset, train_features=BASIC_TRAIN_FEATURES)