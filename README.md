# Moxa A.I. Training Program

This repository provides source code for guide reading course

it's constructed as following:

## tutorial/DNN

In this tutorial, we are going to practice DNN to solve notMNIST dataset. This dataset is designed to look like classic MNIST dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.

There are three parts in this tutorial:

* Download dataset and pre-processing for training and testing 
* Construct DNN with gradient descent
* Construct DNN with stochastic gradient descent

For more information about MNIST, please reference [here](http://yann.lecun.com/exdb/mnist/)  
For more information about notMNIST, please reference [here](http://yaroslavvb.blogspot.tw/2011/09/notmnist-dataset.html)  
All of the source code is reference from [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity)  


## tutorial/CNN

The tuorial is modified from https://github.com/Hvass-Labs/TensorFlow-Tutorials

The major tutorial file is [tutorial.ipynb](https://github.com/dashmoment/moxa_ai_training/blob/master/tutorial/02_CNN/tutorial.ipynb)

You may need to install [ipython Notebook](http://jupyter.org/)

In this turtorial, we will show composition of CIFAR10 dataset and some preprocess for input image at first.

Then, demostrate how to build up a CNN network and train it.

Finally, we will review the training results and some parameters attributes after training 



## tutorial/RNN

In this tutorial, we are going to build LSTM character model. The dataset we use in this practice is Text8.

There are two parts in this tutorial:
* Create word2vector 
* Build LSTM char-level model to generate text

For more information about Text8, please reference [here](http://mattmahoney.net/dc/textdata)   

## CNN Home work solution
There are four main module in this solution
### train.py
Input cifar10 dataset --> build model --> build loss function and log --> training
### model_zoo.py
This script includes two network: ResNet and GoogleLeNet
You can use model ticket to build these networks by calling:
    mz = model_zoo.build_model(#model ticket)
    network = mz.build_model()
### netfactory.py
Common module for building a CNN network. 

### eval.py
The script for evaluation trained model on whole cifar10 test set

