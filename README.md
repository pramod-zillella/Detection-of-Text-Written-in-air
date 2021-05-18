## Detection-of-Text-Written-in-air
Our project involves the recognition of one single digit that is written on the air. To achieve this, we have adopted a deep learning-based approach. The model adopted is a convolutional neural network (CNN) which is most commonly used in the analyses of visual imagery. With this project, we got an accuracy of 98% on the MNIST dataset.
Analyzing the code step by step can make you understand the exact working & process followed.

## Implementation
<p align="center">
<img src="https://user-images.githubusercontent.com/63542593/118702389-7c680700-b832-11eb-86eb-60306a02c104.png" width="600">
</p>

## Network Architecture
<p align="center">
<img src="https://user-images.githubusercontent.com/63542593/118702470-96a1e500-b832-11eb-85e0-f487923b6485.png" width="600">
</p>

## Test Data Creation

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
