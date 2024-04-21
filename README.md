# BertTCR
A deep learning framework for prediction of cancer-related immune states using T cell receptor repertoires. In short, the framework contains two classifiers, including a binary classification model and a multi-class classification model, and their corresponding strategies for immune status assessment.

# Workflow
## ![image name](https://github.com/zhangbeibei-min/BertTCR/tree/main/Figures)


# Installation
## **[link](git clone https://github.com/zhangbeibei-min/BertTCR.git)**



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