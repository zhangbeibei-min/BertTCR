# BertTCR
A deep learning framework for prediction of cancer-related immune status based on T cell receptor repertoires. 


![Workflow ](https://github.com/zhangbeibei-min/BertTCR/tree/main/Workflow)


# Installation

If you want to use BertTCR, please clone github repository on your own machine in a desired folder
```
 git clone https://github.com/zhangbeibei-min/BertTCR.git
 cd BertTCR
```
To run BertTCR, you need to configure the corresponding python package. If they are not installed in your environment, run the following command to install them first:
```
 pip install -r requirements.txt
```

# Usage
## Data Preparation
Before training or prediction, you must convert raw TSV files to .pth format using BERT_embedding.py:

## 1.Using the pre-trained model
 using the pretrained model with the following command:

```
python .\Codes\BertTCR_prediction.py --model_file ./TrainedModels/Pretrained_THCA.pth

```
Prediction results are saved in:./Results/BertTCR_THCA_prediction_result.tsv
Finally, we used./Codes/BertTCR_evalution.py to evaluate performance on test data, including Accuracy, Sensitivity, Specificity, F1-score, AUC and ROC curve, as follows:
```
-----**** [.\BertTCR_THCA_prediction_result.tsv] ****-----
        Accuracy:	0.93
        Sensitivity:	0.875
        Specificity:	0.978
        F1-score:	0.921
        AUC:	0.99
        
```
![ROC](https://github.com/zhangbeibei-min/BertTCR/tree/main/Figures)

## 2.Training a new model
Users can train their own BertTCR model on their own TCR sequencing data samples, train the training set, adjust the parameters of the validation set, and then save the best model of the validation set, using the following command to get better prediction performance:
```
python BertTCR_tranining.py --sample_dir ./TrainingData --val_sample_dir ./validationData --dropout 0.4 --epoch 500 --learning_rate 0.001

```


# Citation 
Please cite our paper if BertTCR is helpful.
Zhang M, Cheng Q, Wei Z, Xu J, Wu S, Xu N, Zhao C, Yu L, Feng W. BertTCR: a Bert-based deep learning framework for predicting cancer-related immune status based on T cell receptor repertoire. Brief Bioinform. 2024 Jul 25;25(5):bbae420. doi: 10.1093/bib/bbae420.