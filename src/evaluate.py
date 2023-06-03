import math, copy

import dgl
import pandas as pd
import numpy as np
import pickle as pkl

import torch as th
import torch.nn as nn
from torch import optim

import time
from easydict import EasyDict

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from utils import get_logger, get_args_from_yaml
from data_generator_ednet import get_dataloader_ednet, get_dataloader_ednet_part
from data_generator_assist import get_dataloader_assist, get_dataloader_assist_part
from data_generator_junyi import get_dataloader_junyi
import config
from train import Evaluator, get_model

from models.igmc import IGMC
from models.dgkt import DGKT
from models.sagkt import SAGKT


def test(args:EasyDict, center_node, logger):
    th.manual_seed(0)
    np.random.seed(0)
    dgl.random.seed(0)

    args.in_feats = config.IN_FEATS
    model = get_model(args)
    print('center_node :', center_node)

    evaluator = Evaluator(args.model_type, args.device, gamma=args.gamma)

    dataloader_manager = DATALOADER_MAP.get(args.dataset)
    train_loader, test_loader = dataloader_manager( 
        args=args,
        data_path=args.dataset,
        batch_size=args.batch_size, 
        num_workers=config.NUM_WORKER,
        seq_len=args.max_seq,
        center_node=center_node
    )

    test_auc, test_acc = evaluator.get_evaluation_result(model, test_loader)

    return test_auc, test_acc

    
import yaml
from collections import defaultdict
from datetime import datetime

DATALOADER_MAP = {
    'assist':get_dataloader_assist,
    'ednet':get_dataloader_ednet,
    'junyi':get_dataloader_junyi,
}

def main():
    with open('./test_configs/test_list.yaml') as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
    file_list = files['files']
    for f in file_list:
        args = get_args_from_yaml(f)
        logger = get_logger(name=args.key, path=f"{args.log_dir}/{args.key}.log")
        logger.info('train args')
        for k,v in args.items():
            logger.info(f'{k}: {v}')

        sub_args = args

        if args.model_type in set(['IGMC', 'SAGKT']) :
            center_node=False
        else: center_node=True

        test_auc, test_acc = test(sub_args, center_node, logger=logger)
        eval_info = {
        'test_auc': test_auc,
        'test_acc': test_acc,
        }
        logger.info('=== test auc {:.4f}, test acc {:.4f} ==='.format(*eval_info.values()))
            
        
if __name__ == '__main__':
    main()