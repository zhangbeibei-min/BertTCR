# BertTCR
A deep learning framework for prediction of cancer-related immune status based on T cell receptor repertoires. This framework utilizes pre-trained protein-BERT models to embed sequences, followed by predictors combining multiple-instance learning (MIL), convolutional neural networks (CNNs), and ensemble learning(EL) to forecast immune status. It includes two classifiers: one for accurately distinguishing between cancer patients and healthy individuals using a binary classification model, and another for discriminating between specific cancer types or healthy status using a multi-class classification model. Additionally, the framework provides corresponding immune state evaluation strategies.For more details, please read our paper.


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
## 1.Using the pre-trained model
In the case of binary classification, the immune status of a healthy individual is classified against the immune status of a THCA patient by first processing the raw data using .\Codes\Prepare_file.py with the following command:
```
python .\Codes\Prepare_file.py  --input_dir .\SampleData\THCA\TestData

```
and then using .\Codes\BERT_embedding.py. The TCR sequence is embedded with the processed data, and then the processed test data is predicted using a Python script, using the pretrained model with the following command:

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

# Contact
Feel free to submit an issue or contact us at 695193839@qq.com for problems about the tool.