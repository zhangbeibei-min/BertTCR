import argparse
import sys
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_recall_curve
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

def create_parser():
    parser = argparse.ArgumentParser(
        description="Script to evaluate the performance of BertTCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        dest="input",
        type=str,
        help="The input prediction file in .tsv format.",
        default='THCA_prediction_result.tsv',
    )
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
    args = parser.parse_args()
    return args


def read_tsv(filename, inf_ind, skip_1st=False, file_encoding="utf8"):
    extract_inf = []
    with open(filename, "r", encoding=file_encoding) as tsv_f:
        if skip_1st:
            tsv_f.readline()
        line = tsv_f.readline()
        while line:
            line_list = line.strip().split("\t")
            temp_inf = []
            for ind in inf_ind:
                temp_inf.append(line_list[ind])
            extract_inf.append(temp_inf)
            line = tsv_f.readline()
    return extract_inf


if __name__ == "__main__":
    # Parse arguments.
    args = create_parser()
    # Read the prediction file.
    prediction_file = read_tsv(args.input, [0, 1, 2], True)

    # Evaluate performance.
    labels, probs, preds = [], [], []
    for sample in prediction_file:
        # Get sample label.
        if sample[0].find(args.flag_positive) != -1:
            labels.append(1)
        elif sample[0].find(args.flag_negative) != -1:
            labels.append(0)
        else:
            try:
                raise ValueError()
            except ValueError as e:
                print("Wrong sample filename! Please name positive samples with '{0}' and negative samples with '{1}'."
                      .format(args.flag_positive, args.flag_negative))
                sys.exit(1)

        # Get probability.
        probs.append(float(sample[1]))
        # Get prediction.
        preds.append(1 if float(sample[1]) > 0.5 else 0)
    # compute metrics.
    acc = round(accuracy_score(labels, preds), 3)
    sen = round(recall_score(labels, preds), 3)
    labels_ = [1 - y for y in labels]
    preds_ = [1 - y for y in preds]
    spe = round(recall_score(labels_, preds_), 3)
    auc = round(roc_auc_score(labels, probs), 3)
    f1 = round(f1_score(labels, preds), 3)
    print('''----- [{0}] -----
        Accuracy:\t{1}
        Sensitivity:\t{2}
        Specificity:\t{3}
        AUC:\t{4}
        F1-score:\t{5}
        '''.format(args.input, acc, sen, spe, auc,f1)
    )
