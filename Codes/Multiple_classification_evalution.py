
import argparse
import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score

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
        default='./Mutiple_classification_prediction.tsv',
    )
    args = parser.parse_args()
    return args

def read_tsv(filename):
    extract_inf = []
    with open(filename, "r", encoding="utf8") as tsv_f:
        next(tsv_f)  # Skip the first line (header)
        for line in tsv_f:
            line_list = line.strip().split("\t")
            probabilities = [float(p) for p in line_list[1:4]]
            prediction = int(float(line_list[4]))
            label = int(line_list[5])
            extract_inf.append((probabilities, prediction, label))
    return extract_inf

if __name__ == "__main__":
    # Parse arguments.
    args = create_parser()
    # Read the prediction file.
    prediction_file = read_tsv(args.input)
    # Separate probabilities, predictions, labels
    probabilities, predictions, labels = zip(*prediction_file)
    # Convert labels and probabilities to numpy arrays
    labels = np.array(labels)
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    # Define class names
    class_names = ['Health', 'BRCA', 'Lung']

    # Compute accuracy for each class
    accuracy = []
    for i, class_name in enumerate(class_names):
        class_labels = np.where(labels == i, 1, 0)
        class_predictions = np.where(predictions == i, 1, 0)
        accuracy.append(accuracy_score(class_labels, class_predictions))
        print("Accuracy ({})".format(class_name), accuracy[i])

    # Compute sensitivity (recall) for each class
    sensitivity = recall_score(labels, predictions, average=None)
    for i, class_name in enumerate(class_names):
        print("Sensitivity ({}): {}".format(class_name, sensitivity[i]))

    # Compute specificity for each class
    specificity = []
    for i, class_name in enumerate(class_names):
        other_labels = np.where(labels != i, 0, 1)  # Non-class samples are considered as "other"
        other_predictions = np.where(predictions != i, 0, 1)  # Non-class predictions are considered as "other"
        tn = np.sum(np.logical_and(other_labels == 0, other_predictions == 0))
        fp = np.sum(np.logical_and(other_labels == 1, other_predictions == 0))  # Corrected calculation
        specificity.append(tn / (tn + fp))
        print("Specificity ({})".format(class_name), specificity[i])

    # Compute F1-score for each class
    f1_scores = f1_score(labels, predictions, average=None)
    for i, class_name in enumerate(class_names):
        print("F1-score ({}): {}".format(class_name, f1_scores[i]))

    # Compute AUC for each class
    auc_scores = roc_auc_score(label_binarize(labels, classes=[0, 1, 2]), probabilities, average=None)
    for i, class_name in enumerate(class_names):
        print("AUC ({}): {}".format(class_name, auc_scores[i]))