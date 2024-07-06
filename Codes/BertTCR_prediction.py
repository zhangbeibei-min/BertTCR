import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.utils.data as Data

# class BertTCR(nn.Module):
#     def __init__(self, filter_num, kernel_size, ins_num, drop_out):
#         super(BertTCR, self).__init__()
#         self.filter_num = filter_num
#         self.kernel_size = kernel_size
#         self.ins_num = ins_num
#         self.drop_out = drop_out
#         self.convs = nn.ModuleList([
#             nn.Sequential(nn.Conv1d(in_channels=768,
#                                     out_channels=filter_num[idx],
#                                     kernel_size=h,
#                                     stride=1),
#                           nn.Sigmoid(),
#                           nn.AdaptiveMaxPool1d(1))
#             for idx, h in enumerate(kernel_size)
#         ])
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(sum(filter_num), 1)
#         self.fc_1 = nn.Linear(ins_num, 2)  # ins_=100，MIL部分
#         self.dropout = nn.Dropout(p=drop_out)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         x = x.reshape(-1, 768, 24)
#         out = [conv(x) for conv in self.convs]
#         out = torch.cat(out, dim=1)#在第二个维度进行拼接
#         out = out.reshape(-1, 1, sum(self.filter_num))
#         out = self.dropout(self.fc(out))#Dropout 是一种正则化技术，用于随机丢弃一部分神经元的输出，以减少过拟合
#         out = out.reshape(-1, self.ins_num)
#         out = self.dropout(self.fc_1(out))
#         out = self.sigmoid(out)
#         return out
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
        #self.fc_1 = nn.Linear(ins_num, 2)  # ins_=100，MIL部分
        self.models = nn.ModuleList([ nn.Linear(ins_num, 2) for _ in range(5)])
        self.dropout = nn.Dropout(p=drop_out)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.reshape(-1, 768, 24)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)#在第二个维度进行拼接
        out = out.reshape(-1, 1, sum(self.filter_num))
        out = self.dropout(self.fc(out))#Dropout 是一种正则化技术，用于随机丢弃一部分神经元的输出，以减少过拟合
        out = out.reshape(-1, self.ins_num)
        # out = self.dropout(self.fc_1(out))
        # out = self.sigmoid(out)
        #  # 使用Bagging方法融合多个模型的预测结果
        pred_sum = 0
        for model in self.models:
            pred = self.dropout(model(out))
            pred_sum += pred
        out = self.sigmoid(pred_sum / len(self.models))

        return out


def create_parser():
    parser = argparse.ArgumentParser(
        description="Script to predict samples using BertTCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample_dir",
        dest="sample_dir",
        type=str,
        help="The directory of samples for prediction.",
        default="/data/zhangm/BertTCR/Data/Lung/TestData",
    )
    parser.add_argument(
        "--model_file",
        dest="model_file",
        type=str,
        help="The pretrained model file for prediction in .pth format.",
        default="/data/zhangm/BertTCR/Model/Pretrained_Lung12.1.pth",
    )
    parser.add_argument(
        "--tcr_num",
        dest="tcr_num",
        type=int,
        help="The number of TCRs in each sample.",
        default=100,
    )
    parser.add_argument(
        "--max_length",
        dest="max_length",
        type=int,
        help="The maximum of TCR length.",
        default=24,
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
        default=0.4,
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        help="The device used to make prediction.",
        default="cuda:0",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=str,
        help="Output file in .tsv format.",
        default='/data/zhangm/BertTCR/Result/Lung_prediction12.1.tsv',
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse arguments.
    args = create_parser()

    # Load model.
    model = BertTCR(filter_num=args.filter_num,
                     kernel_size=args.kernel_size,
                     ins_num=args.tcr_num,
                     drop_out=args.dropout).to(torch.device(args.device))

    model.load_state_dict(torch.load(args.model_file))
    model = model.eval()

    # Predict samples.
    sample_dir = args.sample_dir if args.sample_dir[-1] == "/" else args.sample_dir + "/"
    with open(args.output, "w", encoding="utf8") as output_file:
        output_file.write("Sample\tProbability\tPrediction\n")
        for sample_file in os.listdir(sample_dir):
            # 构造样本文件路径
            sample_path = os.path.join(sample_dir, sample_file)
            # Read sample.
            sample = torch.load(sample_path)
            # 转换为指定设备上的 Tensor
            sample = sample.to(torch.device(args.device))

            # Generate input.
            input_matrix = sample

            # Make prediction.
            predict = model(input_matrix)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            pred = True if prob > 0.5 else False

            # Save result.
            output_file.write("{0}\t{1}\t{2}\n".format(sample_file, prob, pred))
    print("The prediction results have been saved to: " + args.output)