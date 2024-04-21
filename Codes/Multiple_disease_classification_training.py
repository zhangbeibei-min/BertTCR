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
        x = x.reshape(-1, 768, 24)##torch.Size([66400, 768, 24])
        out = [conv(x) for conv in self.convs]#torch.Size([66400, 3, 1])torch.Size([66400, 2, 1])torch.Size([66400, 1, 1])
        out = torch.cat(out, dim=1)#torch.Size([66400, 6, 1])
        out = out.reshape(-1, 1, sum(self.filter_num))#torch.Size([66400, 1, 6])
        out = self.dropout(self.fc(out))#torch.Size([66400, 1, 1])
        out = out.reshape(-1, self.ins_num)#torch.Size([664, 100])
        # # 融合多个模型的预测结果
        pred_sum = 0##1
        for classifier in self.classifiers:##1
            pred = self.dropout(classifier(out))##1
            pred_sum += pred##1
        out = pred_sum / len(self.classifiers)##1
        #out = self.dropout(self.classify(out))##2
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
        default="/data/zhangm/BertTCR/Data/MulticlassData3/TrainingData"
    )
    parser.add_argument(
        "--val_sample_dir",
        dest="val_sample_dir",
        type=str,
        help="The directory of validation samples.",
        default="/data/zhangm/BertTCR/Data/MulticlassData3/ValidationData"
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
        default="/data/zhangm/BertTCR/Model/Pretrained_mutiple_disease_3_classification2.pth",
    )
    args = parser.parse_args()
    return args
# 定义数据集类
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
        sample_data = torch.load(sample_path, "cpu")#加CPU
        # 获取样本标签 Get sample label.
        label = int(sample_file.split(self.flag)[0])
        return sample_data, label
if __name__ == "__main__":
    # Parse arguments解析参数.
    args = create_parser()
    # 设置参数
    sample_dir = args.sample_dir if args.sample_dir[-1] == "/" else args.sample_dir + "/"
    val_sample_dir = args.val_sample_dir if args.val_sample_dir[-1] == "/" else args.val_sample_dir + "/"
    # 创建训练集实例
    train_dataset = TCRDataset(sample_dir, args.flag)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)#len(train_dataset)
    # 创建验证集实例
    val_dataset = TCRDataset(val_sample_dir, args.flag)
    val_loader = Data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    # 设定模型Set model.
    model = BertTCR(filter_num=args.filter_num,
                     kernel_size=args.kernel_size,
                     ins_num=args.tcr_num,
                     drop_out=args.dropout,
                     num_classes=args.num_classes).to(torch.device(args.device))
    criterion = nn.CrossEntropyLoss().to(args.device)  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)  # 优化器  
    # 训练模型Training model.
    train_losses = []  # 存储训练集每个epoch的loss值
    train_accs = []  # 存储训练集每个epoch的准确率值
    val_losses = []  # 存储验证集每个epoch的loss值
    val_accs = []  # 存储验证集每个epoch的准确率值
    best_acc = 0.0  # 最佳准确率
    batch_count = 0  # 记录batch数
    epoch_train_loss = 0.0  # 当前epoch的训练集loss累计值
    epoch_train_correct = 0  # 当前epoch的训练集正确预测样本数累计值
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
            # 记录训练集损失和准确率
            epoch_train_loss += loss.item()
            epoch_train_correct += correct
            batch_count += 1
            # 每隔几个batch记录一次
            if batch_count % args.log_interval == 0:
                train_losses.append(epoch_train_loss / args.log_interval)
                train_accs.append(epoch_train_correct / (args.log_interval * len(batch_y)))
                epoch_train_loss = 0.0
                epoch_train_correct = 0
        # 验证模型Validation model.
        model.eval()  # 验证模式
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
        # 输出训练集和验证集的accuracy和loss
        if (epoch + 1) % args.log_interval == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'time =', '{:.2f}'.format(epoch_time), 's', 'train_loss =',
                  '{:.6f}'.format(train_losses[-1]),
                  'train_acc =', '{:.6f}'.format(train_accs[-1]),
                  'val_loss =', '{:.6f}'.format(loss.item()),
                  'val_acc =', '{:.6f}'.format(accuracy))
            # 保存最佳模型
        if accuracy > best_acc and epoch > 10:
            best_acc = accuracy
            #torch.save(model.state_dict(), args.output)
            print("The best model has been saved with accuracy: {:.4f}".format(best_acc))
    # 调整学习率和训练集曲线的长度与验证集曲线相同
    # lr_values[:len(val_losses)]
    import math
    batch_size=args.batch_size
    result = math.ceil(489 / batch_size)
    train_length = len(train_losses)
    val_length = len(val_losses)
    train_losses = train_losses[::result][:val_length]
    train_accs = train_accs[::result][:val_length]
    ####loss,acc随着epoch的变化曲线
    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(range(1, len(train_accs)+1 ), train_accs, label='Train Acc')
    ax1.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    ax1.plot(range(1, len(val_accs)+1 ), val_accs, label='Val Acc')
    ax1.plot(range(1, len(val_losses)+1 ), val_losses, label='Val Loss')
    ax1.set_ylabel('Value')
    ax1.set_title('Training and Validation Curves')
    ax1.grid(True)
    ax1.legend()
    #plt.savefig('/data/zhangm/BertTCR/Picture/MulticlassData3/training_validation_curves_3_mutiple_diease.jpg')  # 保存图像到指定路径
    plt.show()
    