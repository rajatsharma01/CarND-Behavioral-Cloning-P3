# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[Model]: ./examples/model.png
[OriginalImage]: ./examples/OriginalImage.png
[Augmentations]: ./examples/ImageAugmentations.png
[AngleDistribution]: ./examples/SteeringAngleDistribution.png
[MSELoss]: ./examples/MSELoss.png
[RecoveryLeft]: ./examples/RecoveryLeft.gif
[RecoveryRight]: ./examples/RecoveryRight.gif
[CenterDrive]: ./examples/CenterDrive.gif
[Run1]: ./run1.mp4
[Run2]: ./run2.mp4

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb jupyter notebook containing model implementation and training
* model.py plain python version of jupyter notebook
* model.html hmtl version of jupyter notebook
* drive.py for driving the car in autonomous mode, don't have any changes here, except experimented with speed on fly
* my_model.h5 containing a trained convolution neural network 
* writeup_report.md this file, summarizing the results
* run1.mp4 output of trained model on track1
* run2.mp4 output of trained model on track2

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py my_model.h5
```
Following links to my youtube channel videos are captured with desktop screen session, running simulator.

##### Track1
I had used only Track1 images to train my model, and here is how it works:
[![Track1](https://img.youtube.com/vi/8sW5TuoxRYg/0.jpg)](https://www.youtube.com/watch?v=8sW5TuoxRYg)

![Run1][Run1] is the video created with front camera images stored in run1 dir using video.py utility:
```sh
python video.py run1
```

##### Track2
Following video for Track2 shows, how well the model has adapted to new track, <span style="color: red">never seen it before</span>.
[![Track2](https://img.youtube.com/vi/NbnvLXlP748/0.jpg)](https://www.youtube.com/watch?v=NbnvLXlP748)

![Run2][Run2] is the video created with front camera images stored in run2 dir using video.py utility:
```sh
python video.py run2
```

#### 3. Submission code is usable and readable

The model.ipynb jupyter notebook file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is adapted from NVIDIA's CNN pipeline as described [here](https://arxiv.org/pdf/1604.07316.pdf), consisting of 5 convolution layers: first 3 have 5x5 filter and next 2 have 3x3 filter. There are 3 fully connected (Dense) layers of 100, 50 and 10 depth each and final output layer of depth 1 (single output).

I had to customize NVIDIA's pipeline with following 3 additional layers:
1. First layer to input is a Lamba layer which normalizes the input image x as (x/255.0) - 0.5
2. Second layer is a Croppying2D layer which crops input images by 70 pixels from above and 25 pixels from below.
3. Third layer is a Lamba layer which resizes the images further by 64x64 pixels. This tremendously helps me my model to run    faster by 10x factor, specially on my laptop which doesn't have any GPU.

The model includes ELU layers to introduce nonlinearity, reason for choosing ELU over RELU is to avoid diminishing gradient problem as it doesn't turn off the weights completely, rather sets them to negative values.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting, After each Dense fully connected layer, I have a dropout layer with 0.5 probability. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. To do so, we had split input samples using sklearn.model_selection.train_test_split to split samples into 80% test and 20% validation set. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Model was trained only on track1 images, however when tested on track2, it could stay track througout the run.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. To limit memory consumption of the model, we are using Generators to load images on fly in batch size of 32.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Track1 seems to have more number of left turns only, so training on Track1 can introduce bias for left turns. To generalize model better, I have also collected samples by driving in revese direction on Track1.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Model Architecture][Model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][CenterDrive]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![Recovery from Left Lane][RecoveryLeft]
![Recovery from Right Lane][RecoveryRight]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![Original Image][OriginalImage]
![Augmented Images][Augmentations]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
