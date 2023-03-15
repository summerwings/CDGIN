import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import  logging

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from utils.dynamic_train import GCN_train
from utils.dynamic_infer import GCN_infer

from net.CDGIN import ModelCDGIN

import argparse
import numpy as np

import wandb

logging.basicConfig(level=logging.DEBUG,
                    filename='train_dynamic.log',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

sweep_configuration = {
    'method': 'bayes'
    }

metric = {
    'name': 'loss',
    'goal': 'minimize'
    }

sweep_configuration['metric'] = metric

parameters_dict = {
        'batch_size':{
            'values': [2]
        },
        'LR': {
            'values': [0.0001]
        },
        'epoch': {
            'values': [100]
        },
        'stat_step': {
            'values':[1]
        },
        'val_step': {
            'values':[1]
        },
        'num_nodes': {
            'values': [376]
        },
        'num_classes': {
            'values': [1]
        },
        'window_size':{
            'values': [50]
        },
        'window_stride':{
            'values': [10]
        },
        'num_heads':{
            'values': [1]
        },
        'num_layers':{
            'values': [2]
        },
    'hidden_dim': {
        'values': [128]
    },
    'sparsity': {
        'values': [30]
    },
    'dropout': {
        'values': [0.5]
    },
    'or_regular': {
        'values': [0.0001]
    },
    'cl': {
        'values': [0.1]
    },
}

sweep_configuration['parameters'] = parameters_dict

def cv():

    wandb.init()
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_roc = []
    val_accu = []
    val_SE = []
    val_SP = []
    test_roc = []
    test_accu = []
    test_SE = []
    test_SP = []
    fold_n = 4
    for fold in range(1, fold_n+1):

        logging.info('Fold:{}'.format(fold))

        train_data = r'data/train_data_' + str(fold) + '.npy'
        train_label = r'data/train_label_' + str(fold) + '.npy'
        test_data = r'data/val_data_' + str(fold) + '.npy'
        test_label = r'data/val_label_' + str(fold) + '.npy'


        # loss
        criterion = nn.BCELoss()

        # model
        net = ModelCDGIN(
                input_dim=config.num_nodes,
                hidden_dim=config.hidden_dim,
                num_classes=config.num_classes,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                sparsity=config.sparsity,
                dropout=config.dropout,)
        net.to(device)

        # train
        train = GCN_train(train_data, train_label, test_data, test_label, net, model_name='CDGIN')
        roc, acc, SE, SP = train.train(config)
        val_roc.append(roc)
        val_accu.append(acc)
        val_SE.append(SE)
        val_SP.append(SP)

        wandb.log({
            'fold': fold,
            'val_roc': roc.item(),
            'val_acc': acc.item(),
            'val_SE': SE.item(),
            'val_SP': SP.item(),
        })

        # test
        test_data = r'data/test_data' + '.npy'
        test_label = r'data/test_label' + '.npy'
        infer = GCN_infer(test_data, test_label, net, model_name='CDGIN')
        roc, acc, SE, SP = infer.infer(config)

        test_roc.append(roc)
        test_accu.append(acc.cpu().numpy())
        test_SE.append(SE)
        test_SP.append(SP)

    best_mean_val_roc = sum(val_roc).item() / fold_n
    best_mean_val_accu = sum(val_accu).item() / fold_n
    best_mean_val_SE = sum(val_SE).item() / fold_n
    best_mean_val_SP = sum(val_SP).item() / fold_n

    print('mean accu: %.3f' % (best_mean_val_accu))
    logging.info('Mean best val roc:{}'.format(best_mean_val_roc))
    logging.info('Mean best val accu:{}'.format(best_mean_val_accu))
    logging.info('Mean best val SE:{}'.format(best_mean_val_SE))
    logging.info('Mean best val SP:{}'.format(best_mean_val_SP))

    test_roc = np.stack(test_roc)
    test_accu = np.stack(test_accu)
    test_SE = np.stack(test_SE)
    test_SP = np.stack(test_SP)

    print('mean')
    best_mean_test_roc = test_roc.mean().item()
    best_mean_test_accu = test_accu.mean().item()
    best_mean_test_SE =  test_SE.mean().item()
    best_mean_test_SP = test_SP.mean().item()

    print('sd')
    best_mean_test_roc_sd = test_roc.std().item()
    best_mean_test_accu_sd = test_accu.std().item()
    best_mean_test_SE_sd = test_SE.std().item()
    best_mean_test_SP_sd = test_SP.std().item()

    print('mean accu: %.3f' % (best_mean_val_accu))
    logging.info('Mean best val roc:{}'.format(best_mean_test_roc))
    logging.info('Mean best val accu:{}'.format(best_mean_test_accu))
    logging.info('Mean best val SE:{}'.format(best_mean_test_SE))
    logging.info('Mean best val SP:{}'.format(best_mean_test_SP))

    print('mean accu: %.3f' % (best_mean_val_accu))
    logging.info('Mean best val roc sd:{}'.format(best_mean_test_roc_sd))
    logging.info('Mean best val accu sd:{}'.format(best_mean_test_accu_sd))
    logging.info('Mean best val SE sd :{}'.format(best_mean_test_SE_sd))
    logging.info('Mean best val SP sd:{}'.format(best_mean_test_SP_sd))

    wandb.log({
        'cv_val_roc': best_mean_val_roc,
        'cv_val_acc': best_mean_val_accu,
        'cv_val_SE': best_mean_val_SE,
        'cv_val_SP': best_mean_val_SP,
        'cv_test_roc': best_mean_test_roc,
        'cv_test_acc': best_mean_test_accu,
        'cv_test_SE': best_mean_test_SE,
        'cv_test_SP': best_mean_test_SP,
        'cv_test_roc_sd': best_mean_test_roc_sd,
        'cv_test_acc_sd': best_mean_test_accu_sd,
        'cv_test_SE_sd': best_mean_test_SE_sd,
        'cv_test_SP_sd': best_mean_test_SP_sd,
    })


if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep_configuration, project="Share_Final_DUL")

    wandb.agent(sweep_id, function=cv, count=3)
