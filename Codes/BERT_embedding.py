#!/usr/bin/env python
# encoding: utf-8

from tape import ProteinBertAbstractModel, ProteinBertModel
from tape.models.modeling_utils import SimpleMLP
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection, Mapping
from pathlib import Path
from tape.tokenizers import TAPETokenizer
from tape.datasets import pad_sequences as tape_pad
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import tempfile
from tape import ProteinBertConfig
from torch.utils.data import DataLoader
import pandas as pd
from copy import deepcopy
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import LambdaLR
import logging
import os
import sys
import pickle
import torch.nn.functional as F

logging.basicConfig(format='%(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class BertTCR(ProteinBertAbstractModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = ProteinBertModel(config)
        self.init_weights()
    def forward(self, input_ids, input_mask=None, targets=None):
        outputs = self.bert(input_ids, input_mask=input_mask)
        return outputs
#outputs[0]是最后一层的隐藏状态表示，维度为[batch_size, sequence_length, hidden_size]。
# 它包含了输入序列中每个位置的隐藏状态，可以用于下游任务的特征提取。
# outputs[1]是最后一层的池化表示，维度为[batch_size, hidden_size]。
# 它通过对最后一层的隐藏状态进行池化操作得到，可以用于整体序列的表示。
def create_parser():
    parser = argparse.ArgumentParser(
        description="Script to BertTCR with  data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--eval', type=str,
                        default='/data/zhangm/BertTCR/Data/Lung/TestData.tsv',
                        help='evaluation set')
    parser.add_argument('--train', type=str,
                        default='/data/zhangm/BertTCR/Data/Lung/TrainingData.tsv',
                        help='training set')
    parser.add_argument('--tcrlen', type=int, default=24,
                        help='tcr length')
    parser.add_argument('--epochs', type=int, default=30,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='batch size')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--instance_weight', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='If True, use instance weights from the input data frame')
    args = parser.parse_args()
    return args

class CSVDataset(Dataset):
    def __init__(self,
                 data_file: Union[str, Path, pd.DataFrame],
                 max_tcr_len=24):
        if isinstance(data_file, pd.DataFrame):
            data = data_file
        else:
            data = pd.read_csv(os.path.join(file_path, file_name), delimiter='\t')
        TCR = data['TCR']
        TCR = TCR.apply(lambda x: x[:max_tcr_len])
        self.TCR = TCR.values
        self.data = data
        if 'instance_weights' in data:
            self.instance_weights = data['instance_weights'].values
        else:
            self.instance_weights = np.ones(data.shape[0], )

    def __len__(self) -> int:
        return len(self.TCR)

    def __getitem__(self, index: int):
        seq = self.TCR[index]
        return {
            "id": str(index),
            "primary": seq,
            "tcr_length": len(seq),
            "instance_weights": self.instance_weights[index]
        }

class BertDataset(Dataset):
    def __init__(self,
                 input_file,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 max_tcr_len=24,
                 instance_weight: bool = False):
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self.data = CSVDataset(input_file,
                               max_tcr_len=max_tcr_len, )
        self.instance_weight = instance_weight

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        ret = {'input_ids': token_ids,
               'input_mask': input_mask}
        if self.instance_weight:
            ret['instance_weights'] = item['instance_weights']
        return ret

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        elem = batch[0]
        batch = {key: [d[key] for d in batch] for key in elem}
        input_ids = torch.from_numpy(tape_pad(batch['input_ids'], 0))
        
        input_mask = torch.from_numpy(tape_pad(batch['input_mask'], 0))

        ret = {'input_ids': input_ids,
               'input_mask': input_mask}
        if self.instance_weight:
            instance_weights = torch.tensor(batch['instance_weights'],
                                            dtype=torch.float32)
            ret['instance_weights'] = instance_weights
        return ret

if __name__ == "__main__":
    # Parse arguments.
    args = create_parser()
    torch.manual_seed(args.seed)
    args.tcrlen=24 
    file_path = '/data/zhangm/BertTCR/RawData/Top10/THCA/TrainingData'##ValidationData\TestData\TrainingData
    save_path = '/data/zhangm/BertTCR/Data/Top10/THCA/TrainingData'
    
    # 获取文件夹中的所有文件名，并按照排序顺序进行编码
    file_list = sorted(os.listdir(file_path))
    
    # 初始化模型并将其移动到设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertTCR.from_pretrained('bert-base').to(device)

    file_count = 0  # 用于记录文件数量
    for file_name in file_list:
        if file_name.endswith('.tsv'):
            file_count += 1
            args.train = file_name
            trainset = BertDataset(args.train, max_tcr_len=args.tcrlen, instance_weight=args.instance_weight)
            train_data = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=trainset.collate_fn)
            batch_count = 0  # 用于记录批次数量
            for batch in train_data:
                batch_count += 1
                input_ids = batch['input_ids'].to(device)
                input_mask = batch['input_mask'].to(device)
                outputs = model.forward(input_ids, input_mask)
                
                outputs_list = list(outputs)
                outputs_list[0] = torch.transpose(outputs[0], 1, 2)  # 减缓交换第二维第三维（100,768,24）
                target_n = 24  # 目标维度n为24

                if outputs_list[0].shape[2] > target_n:
                    # 截取前target_n维度 
                    outputs_list[0] = outputs_list[0][:, :, :target_n]
                elif outputs_list[0].shape[2] < target_n:
                    pad_size = (0, target_n - outputs_list[0].shape[2])        # 后面填充0
                    outputs_list[0] = F.pad(outputs_list[0], pad_size, mode='constant', value=0)

                outputs = tuple(outputs_list)

                # 根据读入的文件名+trans命名保存数据
                save_file_name = file_name.replace('.tsv', '_trans.pth')
                save_file_path = os.path.join(save_path, save_file_name)
                torch.save(outputs[0], save_file_path)
                print(outputs[0].shape)
                print(file_name)
                print(f'处理第{file_count}个文件，第{batch_count}个批次')

# #****************************1.Patient和Health两个目录进行编码保存********************
# if __name__ == "__main__":
#     # Parse arguments.
#     args = create_parser()
#     torch.manual_seed(args.seed)
#     args.tcrlen=24 
#     #**********************数据及文件所在路径*************************
#     #1.THCA训练集
#     # file_path = '/data/zhangm/DeepLION/Data/THCA/TrainingData/'  # 文件所在文件夹路径344
#     # save_path = '/data/zhangm/BertTCR/Data/THCA/TrainingData/'  # 保存文件的文件夹路径
#     #2.THCA测试集
#     #file_path = '/data/zhangm/DeepLION/Data/THCA/TestData/'  # 文件所在文件夹路径86
#     #save_path = '/data/zhangm/BertTCR/Data/THCA/TestData/'  # 保存文件的文件夹路径
#     #3.Lung训练集
#     # file_path = '/data/zhangm/DeepLION/Data/Lung/TrainingData/'  # 文件所在文件夹路径356
#     # save_path = '/data/zhangm/BertTCR/Data/Lung/TrainingData/'  # 保存文件的文件夹路径
#     #4.Lung测试集
#     # file_path = '/data/zhangm/DeepLION/Data/Lung/TestData/'  # 文件所在文件夹路径88
#     # save_path = '/data/zhangm/BertTCR/Data/Lung/TestData/'  # 保存文件的文件夹路径
#     #5.Lung验证集
#     # file_path = '/data/zhangm/BertTCR/RawData/Lung_Validation/convert_Lung_Validation'
#     # save_path = '/data/zhangm/BertTCR/Data/Lung/ValidationData'
#     ##5.1health
#     # file_path = '/data/zhangm/BertTCR/RawData/Healthtsv'
#     # save_path = '/data/zhangm/BertTCR/Data/Lung/ValidationData'

#     # # #6.THCA验证集
#     # file_path = '/data/zhangm/BertTCR/RawData/THCA_Validation/tumour_tissue/tsv'
#     # save_path = '/data/zhangm/BertTCR/Data/THCA/ValidationData'

#     ##7.health
#     # file_path = '/data/zhangm/BertTCR/RawData/Healthtsv'
#     # save_path = '/data/zhangm/BertTCR/Data/Lung/ValidationData'
#     #
#     ###8.通用模型数据集
#     # file_path = '/data/zhangm/BertTCR/RawData/TCRdb/IndependentData/PRJNA755141_THCA_24'
#     # save_path = '/data/zhangm/BertTCR/Data/Universal/IndependentData/THCA'
#     ##9.
#     file_path = '/data/zhangm/BertTCR/RawData/MulticlassData/TestData'
#     save_path = '/data/zhangm/BertTCR/Data/MulticlassData/TestData'

#     #**********************读入数据集*******************
#     #1.THCA训练集
#     #file_prefixes = ['Health']
#     file_prefixes = ['Patient']
#     #file_prefixes = ['Health','Patient']
#     # file_numbers = {'Health_': [str(i).zfill(3) for i in range(1, 215)],'Patient_': [str(i).zfill(3) for i in range(1, 131)]}
#     #2.THCA测试集
#     #file_numbers = {'Health_': [str(i).zfill(3) for i in range(1, 47)],'Patient_': [str(i).zfill(3) for i in range(1, 41)]}
#     #3.Lung训练集
#     #file_numbers = {'Health_': [str(i).zfill(3) for i in range(1, 210)],'Patient_': [str(i).zfill(3) for i in range(1, 148)]} 
#     #4.Lung测试集
#     #file_numbers = {'Health_': [str(i).zfill(3) for i in range(1, 52)],'Patient_': [str(i).zfill(3) for i in range(1, 38)]}
#     ##5.Lung验证集
#     #file_numbers = {'Health': [str(i) for i in range(1, 64)], 'Patient': [str(i) for i in range(1, 64)]}###Health不是健康的
#     #5.1Health
#     #file_numbers = {'Health': [str(i) for i in range(1, 98)]}###Health健康的,aimosheng艾默生数据前63个****************
#     #6.THCA验证集
#     #file_numbers = {'Patient': [str(i) for i in range(1, 25)]}
#     #7.THCA验证集
#     #file_numbers = {'Health': [f'{i:03}' for i in range(1, 378)]}
#     #********8.通用模型数据集******************************
#     #file_numbers = {'Health': [str(i).zfill(3) for i in range(1, 239)],'Patient': [str(i).zfill(3) for i in range(1, 242)]}#testdata  238/241
#     #file_numbers = {'Health': [str(i).zfill(3) for i in range(1, 974)],'Patient': [str(i).zfill(3) for i in range(1, 937)]}#trainingdata  973/936
#     file_numbers = {'Patient': [f'{i:03}' for i in range(1, 100)]}
#     # 初始化模型并将其移动到设备上
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = BertTCR.from_pretrained('bert-base').to(device)
#     #model = model.to(device)

#     file_count = 0  # 用于记录文件数量
#     for prefix in file_prefixes:
#         file_number_list = file_numbers[prefix]  # 获取当前前缀下的所有文件编号
#         for number in file_number_list:
#             file_name = prefix + number + '.tsv'
#             if os.path.isfile(os.path.join(file_path, file_name)):
#                 file_count += 1
#                 args.train = file_name
#                 trainset = BertDataset(args.train, max_tcr_len=args.tcrlen,instance_weight=args.instance_weight)
#                 train_data = DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True,collate_fn=trainset.collate_fn)
#                 batch_count = 0  # 用于记录批次数量
#                 for batch in train_data:
#                     batch_count += 1
#                     input_ids = batch['input_ids'].to(device)
#                     input_mask = batch['input_mask'].to(device)
#                     outputs = model.forward(input_ids, input_mask)
                    
#                     outputs_list = list(outputs)
#                     outputs_list[0] = torch.transpose(outputs[0], 1, 2)  # 减缓交换第二维第三维（100,768,24）
#                     target_n = 24  # 目标维度n为24
    
#                     if outputs_list[0].shape[2] > target_n:
#                     # 截取前target_n维度 
#                           outputs_list[0] = outputs_list[0][:, :, :target_n]
#                     elif outputs_list[0].shape[2] < target_n:
#                         pad_size = (0, target_n - outputs_list[0].shape[2])        # 后面填充0
#                         outputs_list[0] = F.pad(outputs_list[0], pad_size, mode='constant', value=0)
    
#                     outputs = tuple(outputs_list)

#                     # 根据读入的文件名+trans命名保存数据
#                     save_file_name = file_name.replace('.tsv', '_trans.pth')
#                     save_file_path = os.path.join(save_path, save_file_name)
#                     torch.save(outputs[0], save_file_path)
#                     print(outputs[0].shape)
#                     print(file_name)
#                     #oo= torch.load("././myTensor.pth")                   
#                     print(f'处理第{file_count}个文件，第{batch_count}个批次')


# **************************2.分批次保存文件*****************************************
# if __name__ == "__main__":
#     # Parse arguments.
#     args = create_parser()
#     torch.manual_seed(args.seed)
#     args.tcrlen = 24

#     # 数据及文件所在路径
#     #1.训练集（对应读入数据集的1和2）
#     # file_path = '/data/zhangm/BertTCR/Data/Universal/TrainingData'
#     # save_path = '/data/zhangm/BertTCR/Data/Universal/TrainingData'
#     # #2.测试集（对应读入数据集的3和4）
#     file_path = '/data/zhangm/BertTCR/Data/Universal/TestData'
#     save_path = '/data/zhangm/BertTCR/Data/Universal/TestData'
#     # 读入数据集
#     #1.训练集健康CDR3
#     # file_prefixes = ['Normal']
#     # file_numbers = "CDR3"
#     #2.训练集疾病CDR3
#     # file_prefixes = ['Tumor']
#     # file_numbers = "CDR3"
#     # #3.测试集健康CDR3
#     # file_prefixes = ['Normal']
#     # file_numbers = "CDR3_test"
#     # #4.测试集疾病CDR3
#     file_prefixes = ['Tumor']
#     file_numbers = "CDR3_test"

#     # 初始化模型并将其移动到设备上
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = BertTCR.from_pretrained('bert-base').to(device)
#     # 用于记录文件数量
#     file_count = 0
#     for prefix in file_prefixes:
#         number = file_numbers  # 获取当前前缀下的所有文件编号
#         file_name = prefix + number + '.tsv'
#         if os.path.isfile(os.path.join(file_path, file_name)):
#             file_count += 1
#             args.train = file_name
#             trainset = BertDataset(args.train, max_tcr_len=args.tcrlen, instance_weight=args.instance_weight)
#             train_data = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=trainset.collate_fn)
#             batch_count = 0  # 用于记录批次数量

#             for batch in train_data:
#                 batch_count += 1
#                 input_ids = batch['input_ids'].to(device)
#                 input_mask = batch['input_mask'].to(device)
#                 outputs = model.forward(input_ids, input_mask)

#                 outputs_list = list(outputs)
#                 outputs_list[0] = torch.transpose(outputs[0], 1, 2)  # 减缓交换第二维第三维（100,768,24）
#                 target_n = 24  # 目标维度n为24

#                 if outputs_list[0].shape[2] > target_n:
#                     outputs_list[0] = outputs_list[0][:, :, :target_n]
#                 elif outputs_list[0].shape[2] < target_n:
#                     pad_size = (0, target_n - outputs_list[0].shape[2])  # 后面填充0
#                     outputs_list[0] = F.pad(outputs_list[0], pad_size, mode='constant', value=0)

#                 outputs = tuple(outputs_list)

#                 # 根据读入的文件名+batch批次数命名保存数据
#                 save_file_name = file_name.replace('.tsv', f'_batch{batch_count}_trans.pth')
#                 save_file_path = os.path.join(save_path, save_file_name)

#                 torch.save(outputs[0], save_file_path)

#                 print(f'处理第{file_count}个文件，第{batch_count}个批次')