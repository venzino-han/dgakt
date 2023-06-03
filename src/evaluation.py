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
from data_generator_ednet import get_dataloader, get_dataloader_part
from data_generator_assist import get_dataloader_assist, get_dataloader_assist_part
from data_generator_junyi import get_dataloader_junyi
import config


from models.igmc import IGMC
from models.dgakt import DGAKT
from models.sagkt import SAGKT

    

def evaluate(model, loader, device):
    # Evaluate AUC, ACC
    model.eval()
    val_labels = []
    val_preds = []
    attention_list = []
    graphs = []
    for batch in loader:
        with th.no_grad():
            preds = model(batch[0].to(device))
            attentions = model.attention.cpu()
            attention_list.append(attentions)

        graphs.append(batch[0].cpu())
        labels = batch[1].to(device) 
        val_labels.extend(labels.cpu().tolist())
        val_preds.extend(preds.cpu().tolist())
        if len(val_preds) > 500:
            break
    
    val_auc = roc_auc_score(val_labels, val_preds)
    val_acc = accuracy_score(list(map(round, val_labels)), list(map(round, val_preds)))
    # val_f1 = f1_score(list(map(round,val_labels)), list(map(round,val_preds)))
    return val_auc, val_acc, graphs, val_labels, val_preds, attention_list

def test(args:EasyDict, center_node, logger):
    th.manual_seed(0)
    np.random.seed(0)
    dgl.random.seed(0)

    ### prepare data and set model
    in_feats = config.IN_FEATS
    if args.model_type == 'IGMC':
        model = IGMC(in_feats=in_feats, 
                     latent_dim=args.latent_dims,
                     num_relations=args.num_relations, 
                     num_bases=4, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

    if args.model_type == 'dgakt':
        model = DGAKT(in_nfeats=in_feats,
                     in_efeats=2,
                     latent_dim=args.latent_dims,
                     ).to(args.device)

    if args.model_type == 'SAGKT':
        model = SAGKT(in_nfeats=in_feats,
                     in_efeats=2,
                     latent_dim=args.latent_dims,
                     ).to(args.device)

    if args.parameters is not None:
        model.load_state_dict(th.load(f"./parameters/{args.parameters}"))

    print('center_node :', center_node)

    dataloader_manager = DATALOADER_MAP.get(args.dataset)
    train_loader, test_loader = dataloader_manager( 
                                                    data_path=args.dataset,
                                                    batch_size=1, 
                                                    num_workers=config.NUM_WORKER,
                                                    seq_len=args.max_seq,
                                                    center_node=center_node
                                                    )

    val_auc, val_acc, graphs, val_labels, val_preds, attention_list = evaluate(model, test_loader, args.device)

    with open('./ednet_graphs.pkl', 'wb') as f:
        pkl.dump(graphs, f)
    with open('./ednet_val_labels.pkl', 'wb') as f:
        pkl.dump(val_labels, f)
    with open('./ednet_val_preds.pkl', 'wb') as f:
        pkl.dump(val_preds, f)
    with open('./ednet_attentions.pkl', 'wb') as f:
        pkl.dump(attention_list, f)

    return

    
import yaml
from collections import defaultdict
from datetime import datetime

DATALOADER_MAP = {
    'assist':get_dataloader_assist,
    'assist_part':get_dataloader_assist_part,
    'assist_part2':get_dataloader_assist_part,
    'ednet':get_dataloader,
    'ednet_part':get_dataloader_part,
    'ednet_part2':get_dataloader_part,
}

def main():
    with open('./test_configs/test_list.yaml') as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
    file_list = files['files']
    for f in file_list:
        date_time = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        args = get_args_from_yaml(f)
        logger = get_logger(name=args.key, path=f"{args.log_dir}/{args.key}.log")
        logger.info('train args')
        for k,v in args.items():
            logger.info(f'{k}: {v}')

        best_lr = None
        sub_args = args
        best_auc_list = []

        
        if args.model_type in set(['IGMC', 'SAGKT']) :
            center_node=False
        else: center_node=True



        best_auc = test(sub_args, center_node, logger=logger)
 
        
if __name__ == '__main__':
    main()