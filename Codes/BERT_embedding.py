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
def create_parser():
    parser = argparse.ArgumentParser(
        description="Script to BertTCR with  data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--eval', type=str,
                        default='./THCA/TestData.tsv',
                        help='evaluation set')
    parser.add_argument('--train', type=str,
                        default='./THCA/TrainingData.tsv',
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
    file_path = './TrainingData'##floder name ValidationData\TestData\TrainingData
    save_path = './TrainingData'
    
    # Gets all the file names in the folder and encodes them in sort order
    file_list = sorted(os.listdir(file_path))
    
    # Initializes the model and moves it to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertTCR.from_pretrained('bert-base').to(device)

    file_count = 0  # Record the number of files
    for file_name in file_list:
        if file_name.endswith('.tsv'):
            file_count += 1
            args.train = file_name
            trainset = BertDataset(args.train, max_tcr_len=args.tcrlen, instance_weight=args.instance_weight)
            train_data = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=trainset.collate_fn)
            batch_count = 0  # Used to record the number of batches

            for batch in train_data:
                batch_count += 1
                input_ids = batch['input_ids'].to(device)
                input_mask = batch['input_mask'].to(device)
                outputs = model.forward(input_ids, input_mask)
                

