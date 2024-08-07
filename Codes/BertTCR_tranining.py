#!/usr/bin/env python
# encoding: utf-8
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
import datetime
import time
class BertTCR(nn.Module):
    def __init__(self, filter_num, kernel_size, ins_num, drop_out):
        super(BertTCR, self).__init__()
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.ins_num = ins_num
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=768,
                                    out_channels=filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.Sigmoid(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(kernel_size)
        ])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(sum(filter_num), 1)
        self.models = nn.ModuleList([ nn.Linear(ins_num, 2) for _ in range(5)])#MIL + ensemble learning
        self.dropout = nn.Dropout(p=drop_out)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.reshape(-1, 768, 24)#
        out = [conv(x) for conv in self.convs]#
        out = torch.cat(out, dim=1)#
        out = out.reshape(-1, 1, sum(self.filter_num))#
        out = self.dropout(self.fc(out))#Dropout
        out = out.reshape(-1, self.ins_num)#
        #  # Merge the predictions of multiple models
        pred_sum = 0
        for model in self.models:
            pred = self.dropout(model(out))
            pred_sum += pred
        out = self.sigmoid(pred_sum / len(self.models))
        return out
def create_parser():
    parser = argparse.ArgumentParser(
        description="Script to train BertTCR with training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample_dir",
        dest="sample_dir",
        type=str,
        help="The directory of training samples.",
        default="./TrainingData"
    )
    parser.add_argument(
    "--val_sample_dir",
    dest="val_sample_dir",
    type=str,
    help="The directory of validation samples.",
    default="./validationData"
    )
    parser.add_argument(
        "--tcr_num",
        dest="tcr_num",
        type=int,
        help="The number of TCRs in each sample.",
        default=100,
    )
    parser.add_argument(
        "--kernel_size",
        dest="kernel_size",
        type=list,
        help="The size of kernels in the convolutional layer.",
        default=[2,3,4],
    )
    parser.add_argument(
        "--filter_num",
        dest="filter_num",
        type=list,
        help="The number of the filters with corresponding kernel sizes.",
        default=[3,2,1],
    )
    parser.add_argument(
        "--dropout",
        dest="dropout",
        type=float,
        help="The dropout rate in one-layer linear classifiers.",
        default=0.4
    )
    parser.add_argument(
        "--epoch",
        dest="epoch",
        type=int,
        help="The number of training epochs.",
        default=500,
    )
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        help="The learning rate used to train BertTCR.",
        default=0.001,
    )
    parser.add_argument(
        "--log_interval",
        dest="log_interval",
        type=int,
        help="The fixed number of intervals to print training conditions",
        default=1,
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        help="The device used to train BertTCR.",
        default="cuda:0",
    )#'cpu'
    parser.add_argument(
        "--flag_positive",
        dest="flag_positive",
        type=str,
        help="The flag in patient sample filename.",
        default="Patient",
    )
    parser.add_argument(
        "--flag_negative",
        dest="flag_negative",
        type=str,
        help="The flag in health individual sample filename.",
        default="Health"
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=str,
        help="Output model file in .pth format.",
        default=".TrainedModels/Pretrained_THCA.pth",
    )
    args = parser.parse_args()
    return args
# Define the data set class
class TCRDataset(Data.Dataset):
    def __init__(self, sample_dir, flag_positive, flag_negative):
        self.samples = []
        self.labels = []
        self.sample_dir = sample_dir
        self.samples = os.listdir(sample_dir)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        sample_path = os.path.join(self.sample_dir, self.samples[index])  #
        sample_data = torch.load(sample_path)
        if self.samples[index].find(flag_positive) != -1:  #
            label = 1
        elif self.samples[index].find(flag_negative) != -1:  #
            label = 0
        else:
            raise ValueError(
                "Wrong sample filename! Please name positive samples with '{0}' and negative samples with '{1}'.".format(
                    flag_positive, flag_negative))
        return sample_data, label
if __name__ == "__main__":
    # Parse arguments.
    args = create_parser()
    # set parameters
    sample_dir = args.sample_dir if args.sample_dir[-1] == "/" else args.sample_dir + "/"
    flag_positive = args.flag_positive
    flag_negative = args.flag_negative
    # Create a dataset instance (training set)
    dataset = TCRDataset(sample_dir, args.flag_positive, args.flag_negative)
    loader = Data.DataLoader(dataset, batch_size=100, shuffle=True)
    # #Create a dataset instance (validation set)
    val_sample_dir = args.val_sample_dir if args.val_sample_dir[-1] == "/" else args.val_sample_dir + "/"
    val_dataset = TCRDataset(val_sample_dir, args.flag_positive, args.flag_negative)
    val_loader = Data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    # 设定模型Set model.
    model = BertTCR(filter_num=args.filter_num,
                     kernel_size=args.kernel_size,
                     ins_num=args.tcr_num,
                     drop_out=args.dropout).to(torch.device(args.device))
    
    criterion = nn.CrossEntropyLoss().to(args.device)#loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.001)#optimizer
    # Training model.
    lr_values = []
    losses = []  #
    accuracies = []  #
    aucs = [] #
    val_losses = []  #
    val_accuracies = []  #
    val_aucs = [] #
    best_auc = 0.0 #
    for epoch in range(args.epoch):
        start_time = time.time()  # 记录epoch开始时间
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_val_loss = 0.0
        epoch_val_accuracy = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(torch.device(args.device)), batch_y.to(torch.device(args.device))
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            # if (epoch + 1) % args.log_interval == 0:
            #     print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            lr_values.append(optimizer.param_groups[0]['lr'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct = (torch.argmax(pred.data.cpu(), dim=1) == batch_y.cpu()).sum().item()
            accuracy = correct / len(batch_y)
            epoch_loss += loss.item()
            epoch_accuracy += accuracy
        epoch_loss /= len(loader)
        epoch_accuracy /= len(loader)
        accuracies.append(epoch_accuracy)
        losses.append(epoch_loss)  #
        if (epoch + 1) % args.log_interval == 0:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print('train_Epoch:', '%04d' % (epoch + 1), 
                  'train_accuracy = ' , '{:.3f}'.format(epoch_accuracy),
                  'train_loss =', '{:.6f}'.format(epoch_loss))
        with torch.no_grad():
            for val_batch_x, val_batch_y in val_loader:
                val_batch_x, val_batch_y = val_batch_x.to(torch.device(args.device)), val_batch_y.to(torch.device(args.device))
                val_pred = model(val_batch_x)
                val_loss = criterion(val_pred, val_batch_y)
                val_auc = roc_auc_score(val_batch_y.cpu().numpy(), val_pred[:, 1].detach().cpu().numpy())
                val_correct = (torch.argmax(val_pred, dim=1) == val_batch_y).sum().item()
                val_accuracy = val_correct / len(val_batch_y)

                epoch_val_loss += val_loss.item()
                epoch_val_accuracy += val_accuracy
                val_aucs.append(val_auc)
                
            epoch_val_loss /= len(val_loader)
            epoch_val_accuracy /= len(val_loader)
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_accuracy)
            if (epoch + 1) % args.log_interval == 0:
                print('valid_Epoch:', '%04d' % (epoch + 1), 
                    'valid_accuracy = ' , '{:.3f}'.format(epoch_val_accuracy),
                    'valid_loss =', '{:.6f}'.format(epoch_val_loss))
            # save best model
            if val_auc > best_auc and epoch > 10:
                best_auc = val_auc
                torch.save(model.state_dict(), args.output)
                print("The best model has been saved with AUC: {:.4f}".format(best_auc))

        #epoch_time = time.time() - start_time  # 计算epoch时间
        #print('Epoch:', '%04d' % (epoch + 1), 'time =', '{:.2f} seconds'.format(epoch_time))  # 输出epoch时间

