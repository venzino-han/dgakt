"""
pre-processing unseen case
"""

import pandas as pd
import os
import shutil

from preprocess import save_user_item_group, BASIC_TRAIN_FEATURES

    
ASSIST_PART_LIST = ['assist_part_a', 'assist_part_b', 'assist_part_c']

for p in ASSIST_PART_LIST:
    path = f'data/{p}'
    os.makedirs(path, exist_ok=True)


for p in ASSIST_PART_LIST:
    src_path = 'data/assist/questions.csv'  
    dst_path = f'data/{p}/questions.csv'  
    shutil.copy(src_path, dst_path)

import pandas as pd

dataset = 'assist'
train_df = pd.read_csv(f'data/{dataset}/train_df.csv')
test_df = pd.read_csv(f'data/{dataset}/test_df.csv')
df = pd.concat([train_df, test_df])

dataset = 'assist_part_a'
test_df = df.query(f'part%3==0')
train_df = df.query(f'part%3!=0')

print(len(train_df)//1000, len(test_df)//1000)
print(set(train_df.part))
print(set(test_df.part))

n = int(len(train_df)*0.9)
train_df[:n].to_csv(f'data/{dataset}/train_df.csv')
train_df[n:].to_csv(f'data/{dataset}/val_df.csv')
test_df.to_csv(f'data/{dataset}/test_df.csv')


dataset = 'assist_part_b'
test_df = df.query(f'part%2==1')
train_df = df.query(f'part%2==0')

print(len(train_df)//1000, len(test_df)//1000)
print(set(train_df.part))
print(set(test_df.part))

n = int(len(train_df)*0.9)
train_df[:n].to_csv(f'data/{dataset}/train_df.csv')
train_df[n:].to_csv(f'data/{dataset}/val_df.csv')
test_df.to_csv(f'data/{dataset}/test_df.csv')



dataset = 'assist_part_c'

part = 2
test_df = df.query(f'part<{part}')
train_df = df.query(f'part>={part}')

print(len(train_df)//1000, len(test_df)//1000)
print(set(train_df.part))
print(set(test_df.part))

n = int(len(train_df)*0.9)
train_df[:n].to_csv(f'data/{dataset}/train_df.csv')
train_df[n:].to_csv(f'data/{dataset}/val_df.csv')
test_df.to_csv(f'data/{dataset}/test_df.csv')


if __name__=="__main__":
    # train_path = "data/train_30m.csv"
    # ques_path = "data/questions.csv"
    # # be aware that appropriate range of data is required to ensure all questions 
    # # are in the training set, or LB score will be much lower than CV score
    # # Recommend to user all of the data.
    # get_group(data_path='ednet_part2')
    # pre_process(train_path, ques_path, 0, -1, 0.8)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset", action="store", default="all")
    args = parser.parse_args()

    for dataset in ASSIST_PART_LIST:
        train_df = pd.read_csv(f'data/{dataset}/train_df.csv', index_col=0)
        val_df = pd.read_csv(f'data/{dataset}/val_df.csv', index_col=0)
        test_df = pd.read_csv(f'data/{dataset}/test_df.csv', index_col=0)
        save_user_item_group(train_df, val_df, test_df, data_path=dataset, train_features=BASIC_TRAIN_FEATURES)