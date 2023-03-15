import random
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from utils.dynamic_npDataset import npDataset
from utils.bold import process_dynamic_fc
from einops import repeat

from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix
import logging

class GCN_train():

    def __init__(self, train_data_file, train_label_file,
                 test_data_file, test_label_file, model, model_name, seed = 15022023):

        self.train_data_file = train_data_file
        self.train_label_file = train_label_file

        self.test_data_file = test_data_file
        self.test_label_file = test_label_file

        self.model = model
        self.model_name = model_name
        self.seed = seed

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def train(self, config):
        # setting
        self.setup_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_set = npDataset(self.train_data_file, self.train_label_file)
        train_data_loader = DataLoader(train_set,
                                       batch_size=config.batch_size, shuffle=True)

        val_set = npDataset(self.test_data_file, self.test_label_file)
        val_data_loader = DataLoader(val_set,
                                     batch_size=1, shuffle=True)

        criterion = nn.BCELoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr= config.LR)
        self.model.to(device)

        best_val_loss = 100000
        best_epoch = 0
        for epoch in range(config.epoch):

            if epoch - best_epoch == 50:
                break

            # information
            train_output = None
            train_label = None
            train_loss = 0

            val_output = None
            val_label = None
            val_loss = 0

            for b, b_data in enumerate(train_data_loader):
                print('training')
                data = b_data['ROI'].to(device)


                dyn_a, sampling_points = process_dynamic_fc(data, config.window_size, config.window_stride, stat_des= 'Cor')
                dyn_a1, sampling_points = process_dynamic_fc(data, config.window_size, config.window_stride, stat_des='Sim')

                sampling_endpoints = [p+config.window_size for p in sampling_points]
                if b==0: dyn_v = repeat(torch.eye(config.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=config.batch_size)
                if len(dyn_a) < config.batch_size: dyn_v = dyn_v[:len(dyn_a)]
                t = data.permute(1, 0, 2)

                label = b_data['label'].to(device)

                self.model.train()
                optimizer.zero_grad()

                outputs, attention, latent, reg_ortho, g_latent = self.model(dyn_v.to(device), dyn_a.to(device), dyn_a1.to(device),
                                                                   t.to(device), sampling_endpoints)

                loss = criterion(outputs, label.unsqueeze(1).float())  + g_latent * config.cl
                print( 'reg_ortho',  reg_ortho, 'g loss', g_latent)
                print('loss:', loss)

                train_loss += loss
                loss.backward()
                optimizer.step()


                if train_output == None and train_label == None:

                    train_output = outputs
                    train_label = label

                else:
                    train_output = torch.cat((train_output, outputs), axis=0)
                    train_label = torch.cat((train_label, label), axis=0)


            logging.info('EPOCH:{}, loss={}'.format(epoch, train_loss))

            if (epoch + 1) % config.stat_step == 0:  # print every T epoch
                # print(outputs)
                train_output = train_output > 0.5
                train_acc = sum(train_output[:, 0] == train_label) / train_output.size(0)
                print('[%d] training loss: %.3f training batch acc %f' % (epoch + 1, train_loss, train_acc))
                training_loss = 0.0

                logging.info('EPOCH:{}, accu={}'.format(epoch, train_acc))

            if (epoch + 1) % config.val_step == 0:


                for v, v_data in enumerate(val_data_loader):

                    data = v_data['ROI'].to(device)

                    dyn_a, sampling_points = process_dynamic_fc(data, config.window_size, config.window_stride,
                                                                stat_des='Cor')
                    dyn_a1, sampling_points = process_dynamic_fc(data, config.window_size, config.window_stride,
                                                                    stat_des='Sim')
                    sampling_endpoints = [p + config.window_size for p in sampling_points]
                    if v == 0: dyn_v = repeat(torch.eye(config.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points),
                                              b=1)
                    if len(dyn_a) < 1: dyn_v = dyn_v[:len(dyn_a)]
                    t = data.permute(1, 0, 2)

                    label = v_data['label'].to(device)

                    self.model.eval()

                    with torch.no_grad():

                        outputs, attention, latent, reg_ortho, g_latent = self.model(dyn_v.to(device), dyn_a.to(device), dyn_a1.to(device),
                                                                           t.to(device), sampling_endpoints)

                        loss = criterion(outputs, label.unsqueeze(1).float())

                        val_loss += loss

                    if val_output == None and val_label == None:

                        val_output = outputs
                        val_label = label

                    else:
                        val_output = torch.cat((val_output, outputs), axis=0)
                        val_label = torch.cat((val_label, label), axis=0)

                # print(outputs)
                val_roc = roc_auc_score(val_label.cpu(), val_output.cpu())
                logging.info('EPOCH:{}, val ROC={}'.format(epoch, val_roc))


                val_output = val_output >= 0.5
                val_acc = sum(val_output[:, 0] == val_label) / val_output.size(0)
                print('[%d] Val loss: %.3f Val batch acc %f' % (epoch + 1, val_loss, val_acc))
                logging.info('EPOCH:{}, val loss={}'.format(epoch, val_loss))
                logging.info('EPOCH:{}, val accu={}'.format(epoch, val_acc))

                tn, fp, fn, tp = confusion_matrix(val_label.cpu(), val_output.cpu()).ravel()
                se = tp / (tp + fn)
                sp = tn / (tn + fp)

                val_recall = se
                logging.info('EPOCH:{}, val recall={}'.format(epoch, val_recall))
                val_precision = sp
                logging.info('EPOCH:{}, val precision={}'.format(epoch, val_precision))

                if val_loss < best_val_loss:

                    logging.info('Best EPOCH:{}'.format(epoch, val_roc))
                    logging.info('New Best Model')
                    best_epoch = epoch
                    print('New Best Model')
                    best_val_roc = val_roc
                    best_val_accu = val_acc
                    best_val_recall = val_recall
                    best_val_precision = val_precision
                    model_name = self.model_name + '_best.pth'
                    torch.save(self.model.state_dict(), model_name)

                    best_val_loss = val_loss

        return best_val_roc, best_val_accu, best_val_recall, best_val_precision