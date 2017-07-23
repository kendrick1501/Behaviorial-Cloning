# UDACITY Self-Driving Car Nanodegree Program

## Deep Learning - Behavioral Cloning
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

[//]: # (Image References)

[image1]: ./model.png "Model Architecture"
[image2]: ./IMG_examples/img_1.jpg "Track one"
[image3]: ./IMG_examples/img_2.jpg "Track one"
[image4]: ./IMG_examples/img_3.jpg "Track two"
[image5]: ./IMG_examples/img_4.jpg "Track two"

### Model Architecture

My model architecture is illustrated in the following table (code line 101-126):

| Layer (type)         |       Output Shape            |  Param #                        |
|:----------------------:|:-------------------------------:|:------------------------------:|
| Cropping2D         |   80 x 320 x 3 |   0                             |
| Lambda  |          80 x 320 x 3  |   0 |
| Conv2D (RELU activation) 7x7   |         74 x 314 x 16 |      2368 |
| MaxPooling 4x4 | 19 x 79 x 16  |  0 |
| Conv2D (RELU activation) 5x5 |          15 x 75 x 24   |     9624 | 
| MaxPooling 2x2 | 8 x 38 x 24   |      0 |
| Conv2D (RELU activation) 3x3 |  6 x 36 x 32  |    6944 |
| MaxPooling 4x4 | 2 x 9 x 32)  |        0| 
|Flatten     |     576    |       0 |
| Dropout (0.15) |     576  |               0|
| Dense       |       550  |              317350|
| Dropout (0.25) |        550 |                0 |
| Dense    |          115  |             63365 |
| Dropout (0.25) |       115 |              0 |
| Dense       |     1      |           116 |

Total parameters: 399,767.

As it can be seen my model is composed by a cropping layer to trim off the region of interest of the image(i.e., the road lane lines) followed by a Lambda layer for data normalization. Then, three convolutional layers followed by RELU activation functions are implemented to extract the road features. Finally, three dense layers followed by dropout regularization are used to estimate the steering angle while avoiding overfitting of the training data. 

The model architecture includes three dropout layers, each one after the dense layers, to avoid overfitting (code line 121, 123, 125). Besides, the model was trained and validated on different data sets (code line 79-83) and tested on both tracks to ensure the car travels safety on the road.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 130).

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving on both tracks and with opposite direction. The three cameras images were used as data for training and validation procedure.

For details about how I created the training data, see the next section. 

### Model Training Strategy

Following the model architecture used for traffic signal classification, my first attempt was to adjust the final fully connected layer output to one and change the loss function to mean square error, given the variable of interest (i.e., steering angle) is continuous instead of categorical.

After some training and validation, I found this model had low mean square error on the training and validation sets. However, the vehicle was not able to safely complete the track (without felling off the track) despite several attempts with different collected data. After revising some forums, I realized that the image reading function (code line 44, 47, 50) output was represented in BGR space. And so, I needed to change it into RGB space representation. With this mistake corrected, I was able to training the neural network so as to gurantee safe autonomous driving on tracks one and two within certain speed limits (less than 20 mph). 

For the final model architecture, a dropout layer was added after the flatten layer in the original model, so as to further avoid overfitting of data.

The final model architecture is presented in table 1 and it is illustrated in the following figure:

![alt text][image1]

####3. Creation of the Training Set & Training Process

The dataset was generated using center lane driving on both tracks. For each track, three laps: two times in normal direction and one time in opposite direction. The images from the three mounted cameras were also considered in the dataset. Below, some examples of center lane driving.

Track one
![alt text][image2]
![alt text][image3]

Track two
![alt text][image4]
![alt text][image5]

After the collection process, I had 39,450 number of images. This dataset is shuffle and then split into a training set (80%) and a validation set (20%). 

A generator function is used to load batches of the sets iteratively (code line 29) to avoid excesive use of memory during the training and validation processes.

In the model architecture, the loaded images are cropped and normalized before being used by the neural network.

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by low change in the final loss value. I used an adam optimizer so that manually training the learning rate wasn't necessary.
