# import argparse
# import sys
# import numpy as np
# import itertools
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score

# def create_parser():
#     parser = argparse.ArgumentParser(
#         description="Script to evaluate the performance of BertTCR.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument(
#         "--input",
#         dest="input",
#         type=str,
#         help="The input prediction file in .tsv format.",
#         default='/data/zhangm/BertTCR/Result/Mutiple_diease_3_prediction1.tsv',
#     )
#     args = parser.parse_args()
#     return args

# def read_tsv(filename):
#     extract_inf = []
#     with open(filename, "r", encoding="utf8") as tsv_f:
#         next(tsv_f)  # Skip the first line (header)
#         for line in tsv_f:
#             line_list = line.strip().split("\t")
#             probabilities = [float(p) for p in line_list[1:4]]
#             prediction = int(float(line_list[4]))
#             label = int(line_list[5])
#             extract_inf.append((probabilities, prediction, label))
#     return extract_inf

# if __name__ == "__main__":
#     # Parse arguments.
#     args = create_parser()
#     # Read the prediction file.
#     prediction_file = read_tsv(args.input)
#     # Separate probabilities, predictions, labels
#     probabilities, predictions, labels = zip(*prediction_file)
#     # Convert labels and probabilities to numpy arrays
#     labels = np.array(labels)
#     probabilities = np.array(probabilities)
#     # Define class names
#     class_names = ['Health', 'BRCA', 'Lung']
#     # Compute ROC curve and ROC area for each class using "one-vs-all" strategy
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     binary_labels = label_binarize(labels, classes=[0, 1, 2])
#     for i in range(len(class_names)):
#          # Set current class as positive (1), others as negative (0)
#         fpr[i], tpr[i], _ = roc_curve(binary_labels[:, i], probabilities[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#     # Compute micro-average ROC curve and ROC area、
#     fpr["micro"], tpr["micro"], _ = roc_curve(binary_labels.ravel(), probabilities.ravel())
#     micro_auc = auc(fpr["micro"], tpr["micro"])

#     # Compute macro-average ROC curve and ROC area
#     #all_fpr = np.unique(np.concatenate(list(fpr.values())))
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(len(class_names)):
#         mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#     mean_tpr /= len(class_names)
#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     macro_auc = auc(fpr["macro"], tpr["macro"])

#     #Plot ROC curves
#     plt.figure(figsize=(8, 8))
#     plt.plot([0, 1], [0, 1], '--', color='gray', lw=3)
#     plt.plot(fpr["micro"], tpr["micro"],
#              label='micro-average ROC curve (area = {0:0.3f})'.format(micro_auc),
#              color='navy', linestyle=':', linewidth=4)##微平均
#     plt.plot(all_fpr, mean_tpr,
#              label='macro-average ROC curve (area = {0:0.3f})'.format(macro_auc),
#              color='deeppink', linestyle=':', linewidth=4)#宏平均

#     colors = ['aqua', 'cornflowerblue',  'darkorange']
#     for i, color in enumerate(colors):
#         plt.plot(fpr[i], tpr[i], color=color, lw=3,
#                  label='ROC curve of class {0} (area = {1:0.3f})'.format(class_names[i], roc_auc[i]))
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc="lower right")
#     plt.tight_layout()
#     plt.savefig('/data/zhangm/BertTCR/Picture/MulticlassData3/roc_curve_plot_3_classification12.5.svg',dpi=300)  # Save ROC curve plot to specified path
#     plt.show()
#     # Compute confusion matrix
#     cm = confusion_matrix(labels, predictions)
#     # Plot confusion matrix
#     plt.figure(figsize=(8, 8))
#     sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.tight_layout()
#     plt.savefig('/data/zhangm/BertTCR/Picture/MulticlassData3/confusion_matrix_3_classification12.5.svg',dpi=300)
#     plt.show()   

#     from sklearn.metrics import precision_recall_curve, auc
#     # Compute precision and recall for each class
#     precision = dict()
#     recall = dict()
#     pr_auc = dict()
#     for i in range(len(class_names)):
#         binary_labels = np.where(labels == i, 1, 0)  # Set current class as positive (1), others as negative (0)
#         precision[i], recall[i], _ = precision_recall_curve(binary_labels, probabilities[:, i])
#         pr_auc[i] = auc(recall[i], precision[i])
#     # Plot PR curves
#     plt.figure(figsize=(8, 8))
#     colors = ['aqua', 'cornflowerblue',  'darkorange']
#     for i, color in enumerate(colors):
#         plt.plot(recall[i], precision[i], color=color, lw=3,
#              label='PR curve of class {0} (AUC = {1:0.3f})'.format(class_names[i], pr_auc[i]))
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend(loc="lower left")
#     plt.tight_layout()
#     plt.savefig('/data/zhangm/BertTCR/Picture/MulticlassData3/pr_curve_plot_3_classification12.5.svg',dpi=300)  # Save PR curve plot to specified path
#     plt.show()


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
        default='H:/BertTCR/Result/Mutiple_diease_3_prediction1.tsv',
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


# #####sci画图合二为一
# import argparse
# import sys
# import numpy as np
# import itertools
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score
#
# def create_parser():
#     parser = argparse.ArgumentParser(
#         description="Script to evaluate the performance of BertTCR.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument(
#         "--input",
#         dest="input",
#         type=str,
#         help="The input prediction file in .tsv format.",
#         default='H:/BertTCR/Result/Mutiple_diease_3_prediction1.tsv',
#     )
#     args = parser.parse_args()
#     return args
#
# def read_tsv(filename):
#     extract_inf = []
#     with open(filename, "r", encoding="utf8") as tsv_f:
#         next(tsv_f)  # Skip the first line (header)
#         for line in tsv_f:
#             line_list = line.strip().split("\t")
#             probabilities = [float(p) for p in line_list[1:4]]
#             prediction = int(float(line_list[4]))
#             label = int(line_list[5])
#             extract_inf.append((probabilities, prediction, label))
#     return extract_inf
#
# if __name__ == "__main__":
#     # Parse arguments.
#     args = create_parser()
#     # Read the prediction file.
#     prediction_file = read_tsv(args.input)
#     # Separate probabilities, predictions, labels
#     probabilities, predictions, labels = zip(*prediction_file)
#     # Convert labels and probabilities to numpy arrays
#     labels = np.array(labels)
#     probabilities = np.array(probabilities)
#     # Define class names
#     class_names = ['Health', 'BRCA', 'Lung']
#     # Compute ROC curve and ROC area for each class using "one-vs-all" strategy
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     binary_labels = label_binarize(labels, classes=[0, 1, 2])
#     for i in range(len(class_names)):
#          # Set current class as positive (1), others as negative (0)
#         fpr[i], tpr[i], _ = roc_curve(binary_labels[:, i], probabilities[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#     # Compute micro-average ROC curve and ROC area、
#     fpr["micro"], tpr["micro"], _ = roc_curve(binary_labels.ravel(), probabilities.ravel())
#     micro_auc = auc(fpr["micro"], tpr["micro"])
#
#     # Compute macro-average ROC curve and ROC area
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(len(class_names)):
#         mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#     mean_tpr /= len(class_names)
#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     macro_auc = auc(fpr["macro"], tpr["macro"])
#
#     # Plot ROC curves
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#
#     axs[0].plot([0, 1], [0, 1], '--', color='gray', lw=3)
#     axs[0].plot(fpr["micro"], tpr["micro"],
#              label='micro-average ROC curve (area = {0:0.3f})'.format(micro_auc),
#              color='navy', linestyle=':', linewidth=4)##微平均
#     axs[0].plot(all_fpr, mean_tpr,
#              label='macro-average ROC curve (area = {0:0.3f})'.format(macro_auc),
#              color='deeppink', linestyle=':', linewidth=4)#宏平均
#
#     colors = ['aqua', 'cornflowerblue',  'darkorange']
#     for i, color in enumerate(colors):
#         axs[0].plot(fpr[i], tpr[i], color=color, lw=3,
#                  label='ROC curve of class {0} (area = {1:0.3f})'.format(class_names[i], roc_auc[i]))
#     axs[0].set_xlim([0.0, 1.0])
#     axs[0].set_ylim([0.0, 1.05])
#     axs[0].set_xlabel('False Positive Rate')
#     axs[0].set_ylabel('True Positive Rate')
#     #axs[0].set_title('Receiver Operating Characteristic')
#     axs[0].legend(loc="lower right")
#
#     # Compute confusion matrix
#     cm = confusion_matrix(labels, predictions)
#     # Plot confusion matrix
#     sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names, ax=axs[1])
#     #axs[1].set_title("Confusion Matrix")
#     axs[1].set_xlabel("Predicted")
#     axs[1].set_ylabel("True")
#
#     plt.tight_layout()
#     #plt.savefig('/data/zhangm/BertTCR/Picture/MulticlassData3/combined_plot.svg', dpi=300)
#     plt.show()
