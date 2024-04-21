# BertTCR
A deep learning framework for prediction of cancer-related immune status based on pre-trained protein-BERT model using T cell receptor repertoires. The framework comprises two classifiers: one for accurately distinguishing between cancer patients and healthy individuals using a binary classification model, and another for discriminating between specific cancer types or healthy status using a multi-class classification model. Additionally, the framework provides corresponding immune state assessment strategies.

# Workflow
## ![image name](https://github.com/zhangbeibei-min/BertTCR/tree/main/Workflow)


### **If you**
```
 git clone https://github.com/zhangbeibei-min/BertTCR.git
 cd BertTCR
```



#  Requirement
Python 3.7.6
torch    
tape
sklearn 0.22.1
numpy 1.21.6
scipy 1.5.2
pandas 1.0.1

#  Note
1. Three_platform_model.py  : Standardize CBC data
2. Immune_status_cluster.py : Used for immune state clustering of CBC data
3. Immune_status_weight.py : Find the correlation coefficient
4. Immune_status_score.py : Score for immune status
5. Cubic_polynomial_fitting.py : To find the relationship between age and immune status score

The code runs in the order shown above.