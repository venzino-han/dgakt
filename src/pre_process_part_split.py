"""
pre-processing ednet
"""

import time
import pickle
import pandas as pd
from copy import copy


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
    parser.add_argument("-d", "--dataset", dest="dataset", action="store", default="ednet")
    args = parser.parse_args()
    print(args)
    