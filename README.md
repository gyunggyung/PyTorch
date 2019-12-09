![](logo/pytorch-logo.png)

---
[![LICENSE](https://img.shields.io/github/license/newhiwoong/PyTorch?style=flat-square)](https://github.com/newhiwoong/PyTorch/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/newhiwoong/PyTorch?style=flat-square&color=yellow)](https://github.com/newhiwoong/PyTorch/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/newhiwoong/PyTorch?style=flat-square&color=informational)](https://github.com/newhiwoong/PyTorch/network/members)
[![GitHub issues](https://img.shields.io/github/issues/newhiwoong/PyTorch?style=flat-square&color=red)](https://github.com/newhiwoong/PyTorch/issues)

tensorflow와 같이 널리 사용되는 딥러닝 라이브러리 중 하나입니다. PyTorch는 tensorflow나 Keras보다 파이써닉합니다.

## Contents  

### PyTorch Basic

1. [basic](01_basic.ipynb) : torch.Tenosr 사용법, 생성, 연산, 데이터 타입  
2. [variable autograd](02_variable_autograd.ipynb) : Variable 사용법, grad, backward 등  
3. [Linear Regression Models](03_Linear_Regression_Models.ipynb) : Linear Models 제작, Naive Model, Neural net Model  
4. [NonLinear Models](04_NonLinear_Models.ipynb) : NonLinear Models 제작, Activation Function, Make Models, Model Save and Load  
5. [Classification Models](05_Classification_Models.ipynb) : Classifcation Models 제작, 2진 분류, 다중 분류  
6. [Batch Tranining](06_Batch_Training.ipynb) : Data.DataLoader, Batch 사용법  

### Neural Network Basic

1. [Optimizers](07_Optimizers.ipynb) : Optimizer function 성능비교  
2. [NN MNIST](08_NN_MNIST.ipynb) : neural network를 이용한 MNIST 진행  
3. [Deep NN MNIST](09_Deep_NN_MNIST.ipynb) : deep neural network를 이용한 MNIST 진행  
4. [CNN MNIST](10_CNN_MNIST.ipynb) : CNN을 이용한 MNIST  
5. [Dropout](12_Dropout.ipynb) : Dropout 적용, 미적용 비교  
6. [RNN]()

### Vison

1. [CNN CIFAR-10](11_CNN_CIFAR-10-GPU.ipynb) : GPU를 이용한 CIFAR-10 CNN 분류 진행 
2. [Transfer Learning]()
3. [CNN High resolution images]()
4. [DCGAN]()
5. [VAE]()

### NLP

1. [Word Embedding]()
2. [Text classificationw]()
3. [NNLMW]()
4. [Seq2Seq]() : 기계 번역
5. [RNN text generation]()
6. [Reinforcement learning text generation]()
7. [ELMO]()
8. [Transformer]()
9. [BERT]()
10. [GPT-2]()
11. [Transformer-XL]()
12. [XLNet]()

## System requirements

```
pip install -r requirements.txt
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
git clone https://github.com/newhiwoong/PyTorch.git
```