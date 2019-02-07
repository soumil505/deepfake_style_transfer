# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 11:06:49 2018

@author: soumil
"""

import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Dropout, Flatten, Dense, Reshape


train1=cv2.imread('trump.jpg')
train2=cv2.imread('rainbow.jpg')
train_image1=np.zeros((256,256,3))
train_image2=np.zeros((256,256,3))

train_image1[:,:,0]=cv2.resize(train1[:,:,0],(256,256))
train_image1[:,:,1]=cv2.resize(train1[:,:,1],(256,256))
train_image1[:,:,2]=cv2.resize(train1[:,:,2],(256,256))

train_image2[:,:,0]=cv2.resize(train2[:,:,0],(256,256))
train_image2[:,:,1]=cv2.resize(train2[:,:,1],(256,256))
train_image2[:,:,2]=cv2.resize(train2[:,:,2],(256,256))

train_image1 = train_image1.reshape(-1, 256, 256, 3)
train_image2 = train_image2.reshape(-1, 256, 256, 3)

input_img = Input(shape=(256, 256, 3,))
x = Conv2D(2, 3, strides=1, activation='relu', padding='same', data_format='channels_last')(input_img)
x = Conv2D(4, 3, strides=2, activation='relu', padding='same', data_format='channels_last')(x)
x=Flatten()(x)
x=Dense(50, activation='relu')(x)
encoder=Dense(128*128*4, activation='relu')(x)

y=Reshape((128,128,4))(encoder)
#y=Dropout(0.3)(y)
y=Conv2DTranspose(2, 3, strides=2, activation='relu', padding='same', data_format='channels_last')(y)
decoder1=Conv2DTranspose(3, 3, strides=1, activation='relu', padding='same', data_format='channels_last')(y)

z=Reshape((128,128,4))(encoder)
#z=Dropout(0.3)(z)
z=Conv2DTranspose(2, 3, strides=2, activation='relu', padding='same', data_format='channels_last')(z)
decoder2=Conv2DTranspose(3, 3, strides=1, activation='relu', padding='same', data_format='channels_last')(z)

model1=Model(input_img,decoder1)
model2=Model(input_img,decoder2)
model1.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model1.summary()
model2.summary()

for i in range (500):
    print("epoch ",i)
    model1.fit(train_image1+np.random.randint(-5,5,size=np.shape(train_image2)),train_image1,epochs=1)
    model2.fit(train_image2+np.random.randint(-5,5,size=np.shape(train_image1)),train_image2,epochs=1)
model1.save('dfmodel1.h5')
model2.save('dfmodel2.h5')
output1=model1.predict(train_image2)
output2=model2.predict(train_image1)
cv2.imwrite('model1.jpg',output1[0,:,:,:].astype(np.uint8))
cv2.imwrite('model2.jpg',output2[0,:,:,:].astype(np.uint8))
cv2.imshow('output1', output1[0,:,:,:].astype(np.uint8))
cv2.imshow('output2', output2[0,:,:,:].astype(np.uint8))
cv2.waitKey()