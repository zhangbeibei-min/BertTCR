import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.utils.data as Data
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
        # # # 融合多个模型的预测结果
        pred_sum = 0##1
        for classifier in self.classifiers:##1
            pred = self.dropout(classifier(out))##1
            pred_sum += pred##1
        out = pred_sum / len(self.classifiers)##1
        #out = self.dropout(self.classify(out))##2
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
        default="/data/zhangm/BertTCR/Data/MulticlassData3/TestData",#ValidationData  TestData
    )
    parser.add_argument(
        "--model_file",
        dest="model_file",
        type=str,
        help="The pretrained model file for prediction in .pth format.",
        default="/data/zhangm/BertTCR/Model/Pretrained_mutiple_disease_3_classification.pth",
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
        "--num_classes",
        dest="num_classes",
        type=int,
        help="The number of disease categories.",
        default=3,
    )
    parser.add_argument(
        "--kernel_size",
        dest="kernel_size",
        type=list,
        help="The size of kernels in the convolutional layer.",
        default=[2,3,4]
    )
    parser.add_argument(
        "--filter_num",
        dest="filter_num",
        type=list,
        help="The number of the filters with corresponding kernel sizes.",
        default=[3,2,1]
    )
    parser.add_argument(
        "--dropout",
        dest="dropout",
        type=float,
        help="The dropout rate in one-layer linear classifiers.",
        default=0.6,
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
        default='/data/zhangm/BertTCR/Result/Mutiple_diease_3_prediction.tsv',
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
                    drop_out=args.dropout,
                    num_classes=args.num_classes).to(args.device)
    model.load_state_dict(torch.load(args.model_file))
    model = model.eval()
    # Predict samples.
    sample_dir = args.sample_dir if args.sample_dir[-1] == "/" else args.sample_dir + "/"
    with open(args.output, "w", encoding="utf8") as output_file:
        output_file.write("Sample\t" + "\t".join(["Probability_Class_" + str(i) for i in range(args.num_classes)]) + "\tPrediction\tLabel\n")
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
            label = int(sample_file.split('_')[0])
            true_labels = [label]
            
            # 将 predict 从 GPU 移动到 CPU
            predict = predict.cpu()
            
            prediction = torch.nn.functional.softmax(predict, dim=1).detach().numpy()[0]
            prediction = np.append(prediction, np.argmax(prediction))
            
            predictions = [prediction]
            
            # Save result.
            output_file.write("{}\t{}\t{}\t{}\n".format(sample_file, "\t".join(map(str, predictions[0][:-1])), predictions[0][-1], true_labels[0]))
    print("The prediction results have been saved to: " + args.output)
