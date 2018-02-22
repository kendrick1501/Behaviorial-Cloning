"""
Created on Mon May  8 22:06:37 2017

@author: kendrick
"""

#=== Importing required libraries ===#

import numpy as np
import cv2

import csv

from sklearn.utils import shuffle        
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Cropping2D
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
#from keras.utils import plot_model

from pathlib import Path

#===================================#

#========== Functions ==============#

def generator(samples, batch_size=32): # Using images from center, left, and right cameras
    
    num_samples = len(samples)
    correction = 0.215
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                name = data_path+img_path+batch_sample[0].split('/')[-1]
                center_img = cv2.imread(name)
                center_img = cv2.cvtColor(center_img,cv2.COLOR_BGR2RGB) # Correcting image color space representation
                name = data_path+img_path+batch_sample[1].split('/')[-1]
                left_img = cv2.imread(name)
                left_img = cv2.cvtColor(left_img,cv2.COLOR_BGR2RGB)
                name = data_path+img_path+batch_sample[2].split('/')[-1]
                right_img = cv2.imread(name)
                right_img = cv2.cvtColor(right_img,cv2.COLOR_BGR2RGB)
                
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                images.extend([center_img, left_img, right_img])
                angles.extend([steering_center, steering_left, steering_right])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
#===================================#

#======== Data loading =============#

samples = []

data_path = './sim_data/'
img_path = 'IMG/'

with open(data_path+'driving_log.csv', 'r', newline='') as f:
    reader = csv.reader(f, delimiter=',')
    for line in reader:
        samples.append(line)
        
samples = shuffle(samples)
         
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)

validation_generator = generator(validation_samples, batch_size=32)

#===================================#

#======== Model architecture =============#

my_model = Path('./model.h5')

if my_model.is_file():
    
    model = load_model('model.h5')
    
else:

    input_shape = (160, 320, 3)
     
    model = Sequential()
     
    model.add(Cropping2D(cropping=((60,20),(0,0)), input_shape = input_shape))
    model.add(Lambda(lambda x: x / 255 - 0.5))
    
    model.add(Conv2D(16, 
                     kernel_size=(7,7), 
                     activation='relu', 
                     padding='valid',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(4,4), padding='same'))
    model.add(Conv2D(24, 
                     kernel_size=(5,5), 
                     activation='relu',
                     padding='valid'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(32, 
                     kernel_size=(3,3), 
                     activation='relu',
                     padding='valid'))
    model.add(MaxPooling2D(pool_size=(4,4), padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.15))
    model.add(Dense(550))
    model.add(Dropout(0.35))
    model.add(Dense(115))
    model.add(Dropout(0.55))
    model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')

#===================================#

#= Model training and validation =#

model.fit_generator(train_generator, steps_per_epoch = 3*len(train_samples)/32, epochs=3, 
                    validation_data=validation_generator, validation_steps = 3*len(validation_samples)/32)

model.save('model.h5')
#plot_model(model, to_file='model.png')

#===================================#
