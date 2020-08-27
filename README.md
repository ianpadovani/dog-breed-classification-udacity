# Dog Breed Classification App - Udacity Project
Uses transfer learning to create a dog breed classifier from VGG16. Takes in an image of a dog and returns the dog breed or the image of a person and returns what dog that person looks like. Implementation uses PyTorch.

The original GitHub repo for this project can be found [here](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification).

## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

Along with exploring state-of-the-art CNN models for classification and localization, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!

## Datasets

The datasets used in this project can be donloaded here:
 - [Dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
 - [Human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz)
 
## Data Augmentation

The images were augmented when training the models. 
 - Resized to 256px on the shortest side, with the rest of the image scaled to match.
 - Randomly cropped and resized to 224px * 224px.
 - There was a 20% chance of the image being flipped horizontally.
 - Images were randomly roated up to a maximum of 15 degrees.

## The "From-Scratch" Model

This model scored an accuracy of **13%** over **25 epochs**. While this accuracy is low overall, it is impressive considering the simplicity of the architecture compared to the difficulty of a 133 class problem.

Net(
- (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
- (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
- (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- (fc1): Linear(in_features=6272, out_features=512, bias=True)
- (fc2): Linear(in_features=512, out_features=512, bias=True)
- (fc3): Linear(in_features=512, out_features=133, bias=True)
- (dropout): Dropout(p=0.25, inplace=False)
- (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)


## The Transfer Learning Model

This model scored an accuracy of **71%** over **10 epochs**. The output layer of the classifier was replaced and retrained to apply the network to the problem of classifying 133 dog breeds.

This evidently performs far better due to its deeper architecture which was pretrained on a much larger set of images which took all forms.

VGG(
- (features): Sequential(
  - (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - (1): ReLU(inplace)
  - (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - (3): ReLU(inplace)
  - (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  - (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - (6): ReLU(inplace)
  - (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - (8): ReLU(inplace)
  - (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  - (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - (11): ReLU(inplace)
  - (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - (13): ReLU(inplace)
  - (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - (15): ReLU(inplace)
  - (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  - (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - (18): ReLU(inplace)
  - (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - (20): ReLU(inplace)
  - (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - (22): ReLU(inplace)
  - (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  - (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - (25): ReLU(inplace)
  - (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - (27): ReLU(inplace)
  - (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - (29): ReLU(inplace)
  - (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
- )
- (classifier): Sequential(
  - (0): Linear(in_features=25088, out_features=4096, bias=True)
  - (1): ReLU(inplace)
  - (2): Dropout(p=0.5)
  - (3): Linear(in_features=4096, out_features=4096, bias=True)
  - (4): ReLU(inplace)
  - (5): Dropout(p=0.5)
  - (6): Linear(in_features=4096, out_features=133, bias=True)
- )
)

