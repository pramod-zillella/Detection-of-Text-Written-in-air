## Detection-of-Text-Written-in-air
Our project involves the recognition of one single digit that is written on the air. To achieve this, we have adopted a deep learning-based approach. The model adopted is a convolutional neural network (CNN) which is most commonly used in the analyses of visual imagery. With this project, we got an accuracy of 98% on the MNIST dataset.
Analyzing the code step by step can make you understand the exact working & process followed. 

For accessing the interactive notebooks:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Pramod04121999/Detection-of-Text-Written-in-air/HEAD)

## Table of Contents
1. [Introduction](#introduction)
2. [Detection](#detection)
3. [Implementation](#implementation)
4. [Network Architecture](#network-architecture)
    * [Convolution layer](#convolution-layer)
    * [Pooling layer](#pooling-layer)
    * [Fully connected layer](#fully-connected-layer)
5. [Results](#results)
6. [Conclusion](#conclusion)

## Introduction
To track an object with the help of the webcam of a laptop and store the data written with that object. The object used can be fingertip or of any specific color, we opted to track objects with a specific color. Once we get the data from tracking the object we then predict the data written on thin air with the help of trained CNN.  

## Detection
The method in the project involves the detection of red and blue color in the frame and masking frame. Once we attain the masked images we proceed to track the red blob by performing bitwise AND with the previous frame red color masked image. Since the mask of the detected blob is black, bitwise AND would store the previous frames information and add the information of the present frame.

Erasing of the mask to write more information or digits is important. We achieved that with the use of detection of the color blue.We calculate the number of blue blobs detected in each frame and once we detect the blue blob for the first time in the frame, we check if enough information has been written on the masked image, if so then we clear the mask so that we can write more information. Blue color detection is used as an all clear for the screen.


<p float="left">
  <img src="https://user-images.githubusercontent.com/63542593/118702606-c51fc000-b832-11eb-9f8b-c7686113f21a.png" width="400">
  <img src="https://user-images.githubusercontent.com/63542593/118702608-c650ed00-b832-11eb-8e87-f6903f7fd440.png" width="400"> 
</p>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; Frames of Writing digit 1     &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Frames of masked red color writing digit 1

&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;(retaining the information of previous frame)

<p float="left">
  <img src="https://user-images.githubusercontent.com/63542593/118702613-c81ab080-b832-11eb-8888-21798ba9632c.png" width="400">
  <img src="https://user-images.githubusercontent.com/63542593/118702616-c8b34700-b832-11eb-8b7c-28060ea2e40d.png" width="400"> 
</p>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; Frames of Writing digit 2     &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Frames of masked red color writing digit 2

&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;(retaining the information of previous frame)
<p float="left">
  <img src="https://user-images.githubusercontent.com/63542593/118702618-c94bdd80-b832-11eb-877e-7cd38c560e19.png" width="400">
  <img src="https://user-images.githubusercontent.com/63542593/118702620-c9e47400-b832-11eb-89c3-a3349659e7f4.png" width="400"> 
</p>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; Frames of Writing digit 3     &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Frames of masked red color writing digit 3

&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;(retaining the information of previous frame)


## Implementation
<p align="center">
<img src="https://user-images.githubusercontent.com/63542593/118702389-7c680700-b832-11eb-86eb-60306a02c104.png" width="600">
</p>

## Network Architecture
In neural networks, convolutional neural networks is one of the main categories to do image recognition, image classification. Object detections, recognition of faces are some of the areas where CNNs are widely used. A CNN is a deep learning algorithm which can take in an image as input, assign importance to various aspects/ objects in the image and categories the images. We prefer CNNs over feed forward neural networks because for image analysis feed forward
neural networks can be overfitting and hence result in wrong outputs. Whereas CNNs can successfully capture the spatial and temporal dependencies in an image through the application of relevant filters or kernels.

### Convolution layer
Convolution is the first layer to extract features from an input image. Convolution preserves the relationship between pixels by learning image features using small squares of input data. It is a mathematical operation that takes two inputs such as image matrix and a filter or kernel. Convolutional of an image gives a feature map as an output that reduces the size of the image and hence reducing the processing time of the algorithm without compromising the accuracy of the model. Convolution of an image with different filters can perform operations such as edge detection, blur and sharpen by applying filters. Convolution of image can give negative values and that is avoided with the help of rectified linear unit for a non linearity operation (ReLU). The output of this operation is max(0,x), where x represents the pixel value in consideration. The performance of this function is better compared to its counterparts, tanh or sigmoid.

### Pooling layer
Pooling layer helps in the reduction of the number of parameters when the image is too large.
There are three types of pooling. Max pooling, we take the largest element from the rectified
feature map. Average pooling, we take the average of the elements of the specified size of the
matrix. Sum pooling, we take the sum of the elements of the said matrix.

### Fully connected layer
We flatten our matrix into a vector and feed it into a fully connected layer like a neural network.
With fully connected layers, we combine features to create a model. We finally have an
activation function such as softmax or sigmoid to classify the outputs into the respective
categories.

<p align="center">
<img src="https://user-images.githubusercontent.com/63542593/118702470-96a1e500-b832-11eb-85e0-f487923b6485.png" width="600">
</p>

## Results

<p float="left">
  <img src="https://user-images.githubusercontent.com/63542593/118705913-6c522680-b836-11eb-8443-ed615c4fa406.png" width="400">
  <img src="https://user-images.githubusercontent.com/63542593/118705920-6d835380-b836-11eb-9bc6-b3b8af14cc8d.png" width="400"> 
</p>

<p align="center">
  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; training dataset accuracy, validation accuracy and training dataset loss, validation loss with each epoch!

</p>

<p align="center">
  
<img src="https://user-images.githubusercontent.com/63542593/118706756-5bee7b80-b837-11eb-9773-951382fa66b1.png" width="400">

</p>

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Accuracy on the created Test dataset

## Conclusion 
The project can be further improved by tracking the finger tip without any color specific detection. The dataset used in CNN can be augmented data to increase the accuracy in the test cases. The detection of multiple digits/alphabets can be achieved with the implementation of Optical Character Recognition (OCR).
