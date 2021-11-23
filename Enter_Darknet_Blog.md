# Enter_Darknet Blog 
This blog is about our project in the Eklavya Mentorship Program by SRA VJTI.  
Here we have documented our journey through the world of Darknet and we hope that this will be a great starting point for others who wish to explore it!

![Darknet, Intro Image](./assets/darknet-main-pic-name.png)

## About This Project

### Aim
The main aim of the project is to develop a general understanding of the darknet framework and its uses in deep learning and object detection, thereby applying our knowledge to the use of image classification using a CNN model.

### Domains Explored In This Project
* Deep Learning and Convolutional Neural Networks.
* Image Classification and Object Detection.
* Image Processing.
* Computer Vision.
  
### What Is Darknet ?
[Darknet](https://pjreddie.com/darknet/) is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation. You can find the source on [GitHub](https://github.com/AlexeyAB/darknet).

![Darknet Logo](./assets/Darknet_Logo.png "Darknet Logo")

The darknet framework has a variety of uses, the most prominent ones being object detection and image classification.  
It is used to classify images ImageNet challenge or to train a classifier on CIFAR-10 dataset.  

You only look once (YOLO) is a state-of-the-art, real-time object detection system which uses the darknet framework.

![YOLO](./assets/yolo.png "Yolo Comparison Graph")

## Approach
The aim as mentioned above is to effectively classify an image among the specified classes according to the provided data set. The proposed project consists of a model that is well researched and comprises various layers such as convolutional pooling and softmax.  
By using convolutional neural networks we make an effective trainable model using these layers and then supply it to the darknet along with our data set and label files where we can train the model using deep learning. Once our model is effectively trained, it will be able to identify, distinguish and correctly label any image fed to it in its data.

![Classification Example](./assets/approach.png "Approach")

To train the model we pass a config file, a data file including the paths of our data set and labels to a darknet train command. We then use it to generate a weights file. Which is passed along with the configured data file during the testing of any image and applies the correct values in the model. Once the model is trained and an image is passed on to it for testing, it classifies the image according to the output of its final softmax layer and prints the probability or percentage of the image being each of the given labels.

## Setting Up Darknet in Linux
The first thing we need to do is install and set up darknet.  
We will be using a Linux environment for the setup, but if it is not available then setting it up in Google Colab or a virtual machine would also work.

#### Prerequisites:
Install python3 (it's preinstalled on ubuntu).  
Install pip.  
Install cmake using: ```sudo pip install cmake```

#### Commands:
```
git clone https://github.com/AlexeyAB/darknet
cd darknet
mkdir build_release
cd build_release
```
#### Now if you are doing it via GPU (CUDA):
  
Follow [this](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-make).

#### If it's via CPU:

Install openCV first using:
```
sudo apt update
sudo apt install libopencv-dev python3-opencv
```

Then ```cmake ..-DENABLE_CUDA=OFF```

And the final command is ```make -j4```

## Dataset For Classification
### CIFAR-10
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

![Dataset](./assets/dataset.png "Dataset")

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

We will be using this dataset to train and test our models.

### Downloading the dataset:

The python, MATLAB, and binary version can be downloaded from this [link](https://www.cs.toronto.edu/~kriz/cifar.html).

But since we are training through darknet, we will use a mirror of the dataset as we want the pictures in image format.
Follow instructions at this [link](https://pjreddie.com/darknet/train-cifar/) to do so.

Also make the cifar.data file in the cfg folder of the cloned darknet repository from the above, but since we will be using our own config files, we don't need the cifra_small.cfg file from the above link.

### Models