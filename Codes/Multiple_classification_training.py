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
import time
from tape.models.modeling_utils import SimpleMLP

class BertTCR(nn.Module):
    def __init__(self, filter_num, kernel_size, ins_num, drop_out, num_classes):
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
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(kernel_size)
        ])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(sum(filter_num), 1)
        self.classifiers = nn.ModuleList([nn.Linear(ins_num, num_classes) for _ in range(10)])##1
        #self.classify = SimpleMLP(ins_num, 512, num_classes)##2
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = x.reshape(-1, 768, 24)##
        out = [conv(x) for conv in self.convs]#
        out = torch.cat(out, dim=1)#
        out = out.reshape(-1, 1, sum(self.filter_num))#
        out = self.dropout(self.fc(out))#
        out = out.reshape(-1, self.ins_num)#
        # # Merge the predictions of multiple models
        pred_sum = 0##1
        for classifier in self.classifiers:##1
            pred = self.dropout(classifier(out))##1
            pred_sum += pred##1
        out = pred_sum / len(self.classifiers)##1
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
        default="./ValidationData"
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
        default=0.6
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
        "--batch_size",
        dest="batch_size",
        type=int,
        help="The number of training batch sizes.",
        default=100,
    )
    parser.add_argument(
        "--num_classes",
        dest="num_classes",
        type=int,
        help="The number of disease categories.",
        default=3,
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
    )  # 'cpu'
    parser.add_argument(
        "--flag",
        dest="flag",
        type=str,
        help="The flag in sample filename.",
        default="_"
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=str,
        help="Output model file in .pth format.",
        default="./Pretrained_mutiple_classification.pth",
    )
    args = parser.parse_args()
    return args
# Define the data set class
class TCRDataset(Data.Dataset):
    def __init__(self, sample_dir, flag):
        self.samples = []
        self.labels = []
        self.sample_dir = sample_dir
        self.samples = os.listdir(sample_dir)
        self.flag = flag
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        sample_file = self.samples[index]
        sample_path = os.path.join(self.sample_dir, sample_file)
        sample_data = torch.load(sample_path)
        #  Get sample label.
        label = int(sample_file.split(self.flag)[0])
        return sample_data, label
if __name__ == "__main__":
    # Parse arguments.
    args = create_parser()
    # set parameters
    sample_dir = args.sample_dir if args.sample_dir[-1] == "/" else args.sample_dir + "/"
    val_sample_dir = args.val_sample_dir if args.val_sample_dir[-1] == "/" else args.val_sample_dir + "/"
    # Create a training set instance
    train_dataset = TCRDataset(sample_dir, args.flag)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)#len(train_dataset)
    # Create a validation set instance
    val_dataset = TCRDataset(val_sample_dir, args.flag)
    val_loader = Data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    # 设定模型Set model.
    model = BertTCR(filter_num=args.filter_num,
                     kernel_size=args.kernel_size,
                     ins_num=args.tcr_num,
                     drop_out=args.dropout,
                     num_classes=args.num_classes).to(torch.device(args.device))
    criterion = nn.CrossEntropyLoss().to(args.device)  # loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)  # optimizer
    # Training model.
    train_losses = []  #
    train_accs = []  #
    val_losses = []  #
    val_accs = []  #
    best_acc = 0.0  #
    batch_count = 0  #
    epoch_train_loss = 0.0  #
    epoch_train_correct = 0  #
    for epoch in range(args.epoch):
        start_time = time.time()
        model.train()  # 训练模式
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(torch.device(args.device)), batch_y.to(torch.device(args.device))
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct = (torch.argmax(pred, dim=1) == batch_y).sum().item()
            accuracy = correct / len(batch_y)
            # Training set losses and accuracy were recorded
            epoch_train_loss += loss.item()
            epoch_train_correct += correct
            batch_count += 1
            # This is recorded every few batches
            if batch_count % args.log_interval == 0:
                train_losses.append(epoch_train_loss / args.log_interval)
                train_accs.append(epoch_train_correct / (args.log_interval * len(batch_y)))
                epoch_train_loss = 0.0
                epoch_train_correct = 0
        # Validation model.
        model.eval()  # verification mode
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(torch.device(args.device)), batch_y.to(torch.device(args.device))
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_losses.append(loss.item())
                correct = (torch.argmax(pred, dim=1) == batch_y).sum().item()
                accuracy = correct / len(batch_y)
                val_accs.append(accuracy)
        end_time = time.time()
        epoch_time = end_time - start_time
        # accuracy loss
        if (epoch + 1) % args.log_interval == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'time =', '{:.2f}'.format(epoch_time), 's', 'train_loss =',
                  '{:.6f}'.format(train_losses[-1]),
                  'train_acc =', '{:.6f}'.format(train_accs[-1]),
                  'val_loss =', '{:.6f}'.format(loss.item()),
                  'val_acc =', '{:.6f}'.format(accuracy))
            # save best model
        if accuracy > best_acc and epoch > 10:
            best_acc = accuracy
            #torch.save(model.state_dict(), args.output)
            print("The best model has been saved with accuracy: {:.4f}".format(best_acc))