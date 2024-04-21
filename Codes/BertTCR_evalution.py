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
        default='/data/zhangm/BertTCR/Result/Lung_prediction12.1.tsv',
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
    # Return n * m matrix "final_inf" (n is the num of lines, m is the length of list "inf_ind").
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

    #************************plot ROC curve,confusion matrix and PRC curve*****************************************

    #************1.plot ROC curve*************************************
    import numpy as np
    import itertools
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.title('THCA')
    #plt.title('Deeplion_validation_THCA')
    plt.legend(loc="lower right")
    #plt.savefig('/data/zhangm/BertTCR/Picture/roc_curve_plot_THCA.jpg')  # 保存ROC曲线图像到指定路径
    plt.savefig('/data/zhangm/BertTCR/Picture/AACNN/AATHCA_roc_curve_plot.jpg')  # 保存ROC曲线图像到指定路径
   
    # *********2.plot confusion matrix******************************
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('THCA')
    #plt.title('Deeplion_validation_THCA')
    #plt.title('THCA')
    #plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # show plots
    #plt.savefig('/data/zhangm/BertTCR/Picture/confusion_matrix_plot_THCA.jpg')  # 保存混淆矩阵图像到指定路径
    plt.savefig('/data/zhangm/BertTCR/Picture/AACNN/AATHCA_confusion_matrix_plot.jpg')  # 保存混淆矩阵图像到指定路径
    plt.show() 

    #####3.plot PRC curve***************************
    from sklearn.metrics import auc 
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    prc_auc = auc(recall, precision)

    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, label='PRC curve (area = %0.3f)' % prc_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.title('THCA')
    plt.legend(loc="lower right")
    #plt.savefig('/data/zhangm/BertTCR/Picture/AACNN/AATHCA_PRC_curve.jpg')  # 保存PRC曲线图像到指定路径
    plt.show()






