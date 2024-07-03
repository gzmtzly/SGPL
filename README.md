# A Novel Classification Method Based on Stepwise Gradient Penalty for Multi-class Imbalanced Data (SGPL) 
Authors: Bimo Ren, Jingyi Feng, and Linyong Zhou
## Abstract  
A novel loss function based on stepwise gradient penalty has been proposed to address model bias in multi-class imbalanced data classification. The method integrates a power exponential function as a penalty factor into the cross-entropy loss, and matches the corresponding gradient penalty according to the frequency of labels in each class. We analyze the rationality of the method from the perspective of gradients and demonstrate that it is the universal framework of the current mainstream imbalanced data classification loss functions. Finally, various types of multi-class imbalanced datasets, including linearly imbalanced datasets, stepped imbalanced datasets, and datasets with consistent/inconsistent training and test distributions, are constructed for classification experiments on the SVHN, CIFAR-10, and Caltech-101 datasets, respectively. The results on all metrics demonstrate the competitiveness of this method.
## 1. Envoronment settings 
   * OS: Ubuntu 18.04.2  
   * GPU: Geforce RTX 2070 
   * Cuda: 11.1, Cudnn: v10.0.130  
   * Python: 3.7.11  
   * Pytorch: 1.6.0   
## 2. Preparation
### 2.1. Data  
We performed binary and multi-class imbalanced data classification experiments on SVHN, CIFAR10, and Clatcah101. Part of the data is included in the code
## 3. Training
### 3.1. File Description 
The file functions are described as
File | Description
--- | --- 
Train_imbalanced_svhn_multi_class.py | training multi-class imbalanced data classification on SVNH
Train_imbalanced_cifar10_multi_class.py | training multi-class imbalanced data classification on cIFAR10
Train_caltech101.py | training multi-class imbalanced data classification on Caltech101
Loss_Function.py | computing SGPL loss 
Model.py | training models on different dataset     
utils.py | computing metrics
### 3.2. Launch the Training
After downloading the code and corresponding data, execute "python Train_xxxx.py" to launch the training. It should be noted that the code contains part of the dataset on SVHN. So, “Train_imbalanced_svhn_multi_class.py” can be executed directly.


