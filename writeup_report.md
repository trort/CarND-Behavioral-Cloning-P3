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

[image1]: ./image/cnn-architecture.png "Model Visualization"
[image2]: ./image/center_drive.jpg "Driving in the center"
[image3]: ./image/left_recover_1.jpg "Recovery Image"
[image4]: ./image/left_recover_2.jpg "Recovery Image"
[image5]: ./image/left_recover_3.jpg "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. All the required files to be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode with speed set to 30 mph
* `model.h5` containing a trained convolution neural network
* `writeup_report.md` summarizing the results
* `video.mp4` containing the camera view recording of a test run in autonomous mode
* `video_game_view.mp4` containing the recording of the simulator screen during the test run

#### 2. Functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Code file for building the model

The `model.py` file contains the code for processing the train images, training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model architecture

I am using [Nvidia's CNN architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) to predict the steering angle from the front camera images (model.py lines 118-139).

The model has 5 convolution layers and 3 fully connected layers, includes RELU activation layers to introduce nonlinearity, and the data is resized and normalized in the model using a Keras lambda layer (code line 106-116). I also tried to use ELU as activation layers but it only reduced training speed without significant improvement in the training accuracy.

#### 2. Attempts to reduce overfitting in the model

The model uses dropout layers in the fully connected layers in order to reduce overfitting (model.py lines 131-135).

The model was trained and validated on randomly-split data sets to ensure that the model was not overfitting (code line 38). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 138).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from edge of the lane, driving with the reversed direction, and all three cameras in the training data.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use existing convolution neural networks that are proven to work in similar tasks.

My first step was to use a simple linear model on the flattened image to confirm the image processing parts are functioning.

The the convolution neural network is designed following the Nvidia architecture, with only one extra initial step to crop the image from 160 x 320 to 110 x 320 and then resize to 66 x 200.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

To combat the overfitting, I added dropout layers between fully connected layers.

Then I tried to train the model with 20 epochs to check when the validation loss stops to decrease. In general, the validation loss reaches the minimum around 5 epochs with different choices of hyper-parameters. Thus the final model is trained with 5 epochs.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially at sharp corners. To improve the driving behavior in these cases, I assigned lower weights to training data with steer equals to 0.0.

At the end of the process, the vehicle is able to drive autonomously around the track in infinite loops without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 118-139) consisted of a convolution neural network with the following layers and layer sizes:

Here is a visualization of the architecture from Nvidia's paper (note: the flattened layer should have 1152 neurons instead of 1164). I added a cropping layer and a lambda layer to normalized the input and resize images to the desired size.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the edge of the road back to center so that the vehicle would learn to recover from off-center positions. These images show what a recovery looks like starting from the left edge :

![alt text][image3]
![alt text][image4]
![alt text][image5]

To better generalize to different track shapes, I also recorded two laps of driving in the reversed direction.

After the collection process, I had 6154 data points. I then randomly shuffled the data set and put 30% of the data into a validation set.

To help the car keep at the center of the track, I also used the images from the left and right cameras and added a correction to the steering angle. My tests show that a correction of 0.1 works best to keep the car in the center. This method triples the images available for training.

To augment the data sat, I also flipped images and angles randomly in the data generator. This effectively doubles the number of images and helps to ensure equal amount of left-steering samples and right-steering samples.

Another trick I found particularly useful is to assign lower weights to data points with 0 steering angle. In the original data set, only 1585 out of 6154 entries have non-zero steering angle. This was because I was using the keyboard control in the training drive, and only pressed the steer button occasionally as each press resulted to a significant steer. Thus the model trained with my driving samples tends to drive in a straight line even when there is a small curvature and fails to steer enough at sharp corners. By assigning lower weights to entries with 0 steer, the model is able to successfully pass all sharp corners.

I used 70% of the data for training the model and the remaining 30% for validation. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as the validation set loss stoped decreasing and the training set loss became much lower than the validation set. I used an adam optimizer so that manually training the learning rate wasn't necessary.
