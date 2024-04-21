# BertTCR
A deep learning framework for prediction of cancer-related immune status based on T cell receptor repertoires. This framework utilizes pre-trained protein-BERT models to embed sequences, followed by predictors combining multiple-instance learning (MIL), convolutional neural networks (CNNs), and ensemble learning(EL) to forecast immune status. It includes two classifiers: one for accurately distinguishing between cancer patients and healthy individuals using a binary classification model, and another for discriminating between specific cancer types or healthy status using a multi-class classification model. Additionally, the framework provides corresponding immune state evaluation strategies.For more details, please read our paper.

# Workflow
## ![ ](https://github.com/zhangbeibei-min/BertTCR/tree/main/Workflow)


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
## 1.Using the pre-trained model


```
-----**** [.\BertTCR_THCA_prediction_result.tsv] ****-----
        Accuracy:	0.93
        Sensitivity:	0.875
        Specificity:	0.978
        F1-score:	0.921
        AUC:	0.99
        
```
## ![ ](https://github.com/zhangbeibei-min/BertTCR/tree/main/Figures)

## 2.Training a new model
Take binary classification for example.
### train the model

BertTCR_tranining.py set lr=
### test the model
dddf

###  evaluation the model
ddd





# Contact
Feel free to submit an issue or contact us at 695193839@qq.com for problems about the tool.