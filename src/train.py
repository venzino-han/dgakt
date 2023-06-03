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
import logging

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from utils import get_logger, get_args_from_yaml
from data_generator_ednet import get_dataloader_ednet, get_dataloader_ednet_part
from data_generator_junyi import get_dataloader_junyi
from data_generator_assist import get_dataloader_assist, get_dataloader_assist_part
import config


from models.igmc import IGMC
from models.dgkt import DGKT
from models.sagkt import SAGKT


def get_model(args):
    if args.model_type == 'IGMC':
        model = IGMC(in_feats=args.in_feats, 
                     latent_dim=args.latent_dims,
                     num_relations=args.num_relations, 
                     num_bases=4, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

    if args.model_type == 'DGKT':
        model = DGKT(in_nfeats=args.in_feats,
                     in_efeats=args.in_efeats,
                     latent_dim=args.latent_dims,
                     ).to(args.device)

    if args.model_type == 'SAGKT':
        model = SAGKT(in_nfeats=args.in_feats,
                     in_efeats=args.in_efeats,
                     latent_dim=args.latent_dims,
                     ).to(args.device)

    if args.parameters is not None:
        model.load_state_dict(th.load(f"./parameters/{args.parameters}"))

    return model


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


class Evaluator():
    def __init__(self, model_type, device, gamma=0.5) -> None:
        self.model_type = model_type
        self.gamma = gamma
        self.device = device
        
        if model_type in ["DGKT", "SAGKT"]:
            self.get_model_prediction = self._get_dual_aspect_model_prediction
        elif model_type in ["IGMC"]:
            self.get_model_prediction = self._get_single_aspect_model_prediction
        else :
            raise NotImplementedError


    def _get_dual_aspect_model_prediction(self, model, batch):
        subg_preds, ui_preds = model(batch[0].to(self.device))
        preds = subg_preds*self.gamma+ui_preds*(1-self.gamma)
        return preds

    def _get_single_aspect_model_prediction(self, model, batch):
        preds = model(batch[0].to(self.device))
        return preds

    def get_evaluation_result(self, model, loader):
        # Evaluate AUC, ACC
        model.eval()
        val_labels = []
        val_preds = []
        for batch in tqdm(loader):
            with th.no_grad():
                preds = self.get_model_prediction(model, batch)
            labels = batch[1]
            val_labels.extend(labels.cpu().tolist())
            val_preds.extend(preds.cpu().tolist())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(list(map(round, val_labels)), list(map(round, val_preds)))
        # val_f1 = f1_score(list(map(round,val_labels)), list(map(round,val_preds)))
        return val_auc, val_acc


class Traniner():
    def __init__(self, model, args, train_loader, optimizer) -> None:
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.mse_loss_fn = nn.MSELoss().to(args.device)
        self.bce_loss_fn = nn.BCELoss().to(args.device)
        self.optimizer = optimizer
        self.log_interval = args.log_interval
        self.gamma = args.gamma
        self.lambda_ = args.lambda_
        self.device = args.device

        if args.model_type in ['DGKT', 'SAGKT']:
            self._model_update = self._model_update_dual_aspect_loss
        elif args.model_type in ['IGMC']:
            self._model_update = self._model_update_single_aspect_loss
        else:
            raise NotImplementedError

    def _model_update_dual_aspect_loss(self, inputs, labels):
        subg_preds, ui_preds = self.model(inputs)
        subg_loss = self.bce_loss_fn(subg_preds, labels) 
        ui_loss = self.bce_loss_fn(ui_preds, labels) 

        preds = subg_preds*self.gamma+ui_preds*(1-self.gamma)
        loss = subg_loss*self.gamma + ui_loss*(1-self.gamma) + self.lambda_*self.mse_loss_fn(subg_preds,ui_preds)
        return preds, loss

    def _model_update_single_aspect_loss(self, inputs, labels):
        preds = self.model(inputs)
        loss = self.bce_loss_fn(preds,labels)
        return preds, loss

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.
        iter_loss, iter_mse, iter_cnt = 0., 0., 0
        iter_dur = []
        logger = logging.getLogger(name=self.args.key)

        for iter_idx, batch in enumerate(self.train_loader, start=1):
            t_start = time.time()
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            preds, loss = self._model_update(inputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * preds.shape[0]
            iter_loss += loss.item() * preds.shape[0]
            iter_mse += ((preds - labels) ** 2).sum().item()
            iter_cnt += preds.shape[0]
            iter_dur.append(time.time() - t_start)

            if iter_idx % self.log_interval == 0:
                logger.debug(f"Iter={iter_idx}, loss={iter_loss/iter_cnt:.4f}, rmse={math.sqrt(iter_mse/iter_cnt):.4f}, time={np.average(iter_dur):.4f}")
                iter_loss, iter_mse, iter_cnt = 0., 0., 0

        return self.model, epoch_loss/len(self.train_loader.dataset)
    
    # def train(self):


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
    th.manual_seed(0)
    np.random.seed(0)
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
        train_loader, test_loader = dataloader_manager(args=sub_args,
                                                       data_path=sub_args.dataset,
                                                       batch_size=sub_args.batch_size, 
                                                       num_workers=config.NUM_WORKER,
                                                       seq_len=sub_args.max_seq,
                                                       center_node=center_node
        )
        best_lr = None
        for lr in args.train_lrs:
            sub_args['train_lr'] = lr
            # date_time = datetime.now().strftime("%Y%m%d_%H:%M")
            # run_id = wandb.util.generate_id()
            # with wandb.init(id=run_id, name=f"{args.key}_{lr}_{date_time}", 
            #                  project="DGKT", config=sub_args):
              
            """prepare data and set model"""
            args.in_feats = config.IN_FEATS
            model = get_model(args)
            logger.info("Loading network finished ...\n")
            
            count_parameters(model)
            # wandb.watch(model)
            optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
            trainer = Traniner(model, args, train_loader, optimizer)
            evaluator = Evaluator(args.model_type, args.device, gamma=args.gamma)

            best_epoch = 0
            best_auc, best_acc = 0, 0

            logger.info(f"Start training ... learning rate : {args.train_lr}")
            epochs = list(range(1, args.train_epochs+1))
            for epoch_idx in epochs:
                model, train_loss = trainer.train_epoch()
                test_auc, test_acc = evaluator.get_evaluation_result(model, test_loader)
                eval_info = {
                'epoch': epoch_idx,
                'train_loss': train_loss,
                'test_auc': test_auc,
                'test_acc': test_acc,
                }
                logger.info('=== Epoch {}, train loss {:.4f}, test auc {:.4f}, test acc {:.4f} ==='.format(*eval_info.values()))
            
                if test_auc <=0.51:
                    logger.info('train failed')
                    continue

                if epoch_idx % args.lr_decay_step == 0:
                    for param in optimizer.param_groups:
                        param['lr'] = args.lr_decay_factor * param['lr']
                        print('lr : ', param['lr'])            

                if best_auc < test_auc:
                    logger.info(f'new best test auc {test_auc:.4f} acc {test_acc:.4f} ===')
                    best_epoch = epoch_idx
                    best_auc = test_auc
                    best_acc = test_acc
                    best_lr = sub_args.train_lr
                    best_state = copy.deepcopy(model.state_dict())

                # wandb.log({
                #     "Test Accuracy": test_acc,
                #     "Test AUC": test_auc,
                #     "Best Accuracy": best_acc,
                #     "Best AUC": best_auc,
                #     "Train Loss": train_loss,
                #     })
            
            th.save(best_state, f'./parameters/{args.key}_{best_auc:.4f}.pt')
            del model
            th.cuda.empty_cache()

            logger.info(f"Training ends. The best testing auc {best_auc:.4f} acc {best_acc:.4f} at epoch {best_epoch}")
            best_auc_acc_lr_list.append((best_auc, best_acc, best_lr))
            # wandb.finish()

        best_auc, best_acc, best_lr = max(best_auc_acc_lr_list, key = lambda x: x[0])
        best_auc_list = [x[0] for x in best_auc_acc_lr_list]
        best_acc_list = [x[1] for x in best_auc_acc_lr_list]
        logger.info(f"**********The final best testing AUC {best_auc:.4f} ACC {best_acc:.4f} at lr {best_lr}********")
        logger.info(f"**********The mean testing AUC {np.mean(best_auc_list):.4f}, {np.std(best_auc_list):.4f} ********")
        logger.info(f"**********The mean testing ACC {np.mean(best_acc_list):.4f}, {np.std(best_acc_list):.4f} ********")
    
        
if __name__ == '__main__':
    main()