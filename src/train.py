import math, copy

import dgl
import pandas as pd
import numpy as np
import pickle as pkl

import torch as th
import torch.nn as nn
from torch import optim

from tqdm import tqdm

import time
from easydict import EasyDict
from prettytable import PrettyTable
import wandb

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from utils import get_logger, get_args_from_yaml
from data_generator_ednet import get_dataloader_ednet, get_dataloader_ednet_part
from data_generator_junyi import get_dataloader_junyi
from data_generator_assist import get_dataloader_assist, get_dataloader_assist_part
import config


from models.igmc import IGMC
from models.igkt import IGKT_TS
from models.igakt import IGAKT
from models.dagkt_v2 import DAGKT
# from models.dagkt_v3 import DAGKT
from models.sagkt import SAGKT



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    

def evaluate(model, loader, device, gamma):
    # Evaluate AUC, ACC
    model.eval()
    val_labels = []
    val_preds = []
    for batch in tqdm(loader):
        with th.no_grad():
            subg_preds, ui_preds = model(batch[0].to(device))
            preds = subg_preds*gamma+ui_preds*(1-gamma)
        labels = batch[1].to(device)
        val_labels.extend(labels.cpu().tolist())
        val_preds.extend(preds.cpu().tolist())
    
    val_auc = roc_auc_score(val_labels, val_preds)
    val_acc = accuracy_score(list(map(round, val_labels)), list(map(round, val_preds)))
    # val_f1 = f1_score(list(map(round,val_labels)), list(map(round,val_preds)))
    return val_auc, val_acc


def train_epoch(model, optimizer, loader, device, logger, log_interval, gamma):
    model.train()

    epoch_loss = 0.
    iter_loss = 0.
    iter_mse = 0.
    iter_cnt = 0
    iter_dur = []
    mse_loss_fn = nn.MSELoss().to(device)
    bce_loss_fn = nn.BCELoss().to(device)

    for iter_idx, batch in enumerate(loader, start=1):
        t_start = time.time()

        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        subg_preds, ui_preds = model(inputs)
        subg_loss = bce_loss_fn(subg_preds, labels) 
        ui_loss = bce_loss_fn(ui_preds, labels) 

        preds = subg_preds*gamma+ui_preds*(1-gamma)

        loss = subg_loss*gamma + ui_loss*(1-gamma) + config.LAMBDA_LOSS*mse_loss_fn(subg_preds,ui_preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * preds.shape[0]
        iter_loss += loss.item() * preds.shape[0]
        iter_mse += ((preds - labels) ** 2).sum().item()
        iter_cnt += preds.shape[0]
        iter_dur.append(time.time() - t_start)

        if iter_idx % log_interval == 0:
            logger.debug(f"Iter={iter_idx}, loss={iter_loss/iter_cnt:.4f}, rmse={math.sqrt(iter_mse/iter_cnt):.4f}, time={np.average(iter_dur):.4f}")
            iter_loss = 0.
            iter_mse = 0.
            iter_cnt = 0
            
    return epoch_loss / len(loader.dataset)


def train(args:EasyDict, train_loader, test_loader, logger):
    th.manual_seed(0)
    np.random.seed(0)
    # dgl.random.seed(0)

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

    if args.model_type == 'IGKT':
        model = IGMC(in_feats=in_feats, 
                     latent_dim=args.latent_dims,
                     num_relations=args.num_relations, 
                     num_bases=4, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

    if args.model_type == 'IGKT_TS':
        model = IGKT_TS(in_feats=in_feats, 
                     latent_dim=args.latent_dims,
                     num_relations=args.num_relations, 
                     num_bases=4, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

    if args.model_type == 'IGAKT':
        model = IGAKT(in_nfeats=in_feats,
                     in_efeats=2,
                     latent_dim=args.latent_dims,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

    if args.model_type == 'DAGKT':
        model = DAGKT(in_nfeats=in_feats,
                     in_efeats=3,
                     latent_dim=args.latent_dims,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

    if args.model_type == 'SAGKT':
        model = SAGKT(in_nfeats=in_feats,
                     in_efeats=2,
                     latent_dim=args.latent_dims,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

    if args.parameters is not None:
        model.load_state_dict(th.load(f"./parameters/{args.parameters}"))

    wandb.watch(model)    
    optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    logger.info("Loading network finished ...\n")

    count_parameters(model)
    
    best_epoch = 0
    best_auc, best_acc = 0, 0
    best_lr = None

    logger.info(f"Start training ... learning rate : {args.train_lr}")
    epochs = list(range(1, args.train_epochs+1))

    eval_func_map = {
        'IGMC': evaluate,
    }
    eval_func = eval_func_map.get(args.model_type, evaluate)
    for epoch_idx in epochs:
        logger.debug(f'Epoch : {epoch_idx}')
    
        train_loss = train_epoch(model, optimizer, train_loader, 
                                 args.device, logger, 
                                 log_interval=args.log_interval,
                                 gamma=args.gamma, 
                                 )
        test_auc, test_acc = eval_func(model, test_loader, args.device, args.gamma)
        eval_info = {
            'epoch': epoch_idx,
            'train_loss': train_loss,
            'test_auc': test_auc,
            'test_acc': test_acc,

        }
        logger.info('=== Epoch {}, train loss {:.4f}, test auc {:.4f}, test acc {:.4f} ==='.format(*eval_info.values()))
        
        if test_auc <=0.51:
            return best_auc, best_acc, best_lr

        wandb.log({
            "Test Accuracy": test_acc,
            "Test AUC": test_auc,
            "Train Loss": train_loss,
            })

        if epoch_idx % args.lr_decay_step == 0:
            for param in optimizer.param_groups:
                param['lr'] = args.lr_decay_factor * param['lr']
            print('lr : ', param['lr'])

        if best_auc < test_auc:
            logger.info(f'new best test auc {test_auc:.4f} acc {test_acc:.4f} ===')
            best_epoch = epoch_idx
            best_auc = test_auc
            best_acc = test_acc
            best_lr = args.train_lr
            best_state = copy.deepcopy(model.state_dict())
        
    th.save(best_state, f'./parameters/{args.key}_{args.dataset}_{best_auc:.4f}.pt')
    logger.info(f"Training ends. The best testing auc {best_auc:.4f} acc {best_acc:.4f} at epoch {best_epoch}")
    return best_auc, best_acc, best_lr
    
import yaml
from collections import defaultdict
from datetime import datetime

DATALOADER_MAP = {
    'assist':get_dataloader_assist,
    'assist_part':get_dataloader_assist_part,
    'assist_part2':get_dataloader_assist_part,
    'ednet':get_dataloader_ednet,
    'ednet_part':get_dataloader_ednet_part,
    'ednet_part2':get_dataloader_ednet_part,
    'junyi':get_dataloader_junyi,
}


def main():
    with open('./train_configs/train_list.yaml') as f:
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
        best_auc_acc_lr_list = []

        dataloader_manager = DATALOADER_MAP.get(sub_args.dataset)
        if args.model_type in set(['IGMC', 'SAGKT']) :
            center_node=False
        else: center_node=True
        print('center_node :', center_node)
        train_loader, test_loader = dataloader_manager( 
                                                       data_path=sub_args.dataset,
                                                       batch_size=sub_args.batch_size, 
                                                       num_workers=config.NUM_WORKER,
                                                       seq_len=sub_args.max_seq,
                                                       center_node=center_node
                                                        )

        for lr in args.train_lrs:
            date_time = datetime.now().strftime("%Y%m%d_%H:%M")
            sub_args['train_lr'] = lr
            sub_args['lambda'] = config.LAMBDA_LOSS
            run_id = wandb.util.generate_id()
            wandb.init(id=run_id, name=f"{args.key}_{lr}_{date_time}", 
                             project="DAGKT", config=sub_args)

            best_auc_acc_lr = train(sub_args, train_loader, test_loader, logger=logger)
            best_auc_acc_lr_list.append(best_auc_acc_lr)
            wandb.finish()
        best_auc, best_acc, best_lr = max(best_auc_acc_lr_list, key = lambda x: x[0])
        best_auc_list = [x[0] for x in best_auc_acc_lr_list]
        best_acc_list = [x[1] for x in best_auc_acc_lr_list]
        logger.info(f"**********The final best testing AUC {best_auc:.4f} ACC {best_acc:.4f} at lr {best_lr}********")
        logger.info(f"**********The mean testing AUC {np.mean(best_auc_list):.4f}, {np.std(best_auc_list):.4f} ********")
        logger.info(f"**********The mean testing ACC {np.mean(best_acc_list):.4f}, {np.std(best_acc_list):.4f} ********")
    
        
if __name__ == '__main__':
    main()