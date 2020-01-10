import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def mobileunet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    initializer = {'depthwise_initializer':'he_normal','pointwise_initializer':'he_normal'}
    #conv1  = Conv2D(64,3,activation='relu',padding='same', kernel_initializer='he_normal')(inputs)
    #conv1  = Conv2D(64,3,activation='relu',padding='same', kernel_initializer='he_normal')(conv1)
    conv1  = SeparableConv2D(64, 3, activation='relu', padding='same', **initializer)(inputs)
    conv1  = SeparableConv2D(64, 3, activation='relu', padding='same', **initializer)(conv1)
    conv1  = BatchNormalization()(conv1)
    pool1  = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2  = SeparableConv2D(128, 3, activation='relu', padding='same', **initializer)(pool1)
    conv2  = BatchNormalization()(conv2)
    conv2  = SeparableConv2D(128, 3, activation='relu', padding='same', **initializer)(conv2)
    conv2  = BatchNormalization()(conv2)
    pool2  = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3  = SeparableConv2D(256, 3, activation='relu', padding='same', **initializer)(pool2)
    conv3  = BatchNormalization()(conv3)
    conv3  = SeparableConv2D(256, 3, activation='relu', padding='same', **initializer)(conv3)
    conv3  = BatchNormalization()(conv3)
    pool3  = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4  = SeparableConv2D(512, 3, activation='relu', padding='same', **initializer)(pool3)
    conv4  = BatchNormalization()(conv4)
    conv4  = SeparableConv2D(512, 3, activation='relu', padding='same', **initializer)(conv4)
    conv4  = BatchNormalization()(conv4)
    drop4  = Dropout(0.2)(conv4)
    pool4  = MaxPooling2D(pool_size=(2, 2))(drop4)    
    
    conv5  = SeparableConv2D(1024, 3, activation='relu', padding='same', **initializer)(pool4)
    conv5  = BatchNormalization()(conv5)
    conv5  = SeparableConv2D(1024, 3, activation='relu', padding='same', **initializer)(conv5)
    conv5  = BatchNormalization()(conv5)
    drop5  = Dropout(0.2)(conv5)
    
    up6    = Conv2DTranspose(512, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(drop5)
    merge6 = concatenate([drop4, up6], axis = 3)
    conv6  = SeparableConv2D(512, 3, activation='relu', padding='same', **initializer)(merge6)
    conv6  = BatchNormalization()(conv6)
    conv6  = SeparableConv2D(512, 3, activation='relu', padding='same', **initializer)(conv6)
    conv6  = BatchNormalization()(conv6)
    
    up7    = Conv2DTranspose(256, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7  = SeparableConv2D(256, 3, activation='relu', padding='same', **initializer)(merge7)
    conv7  = BatchNormalization()(conv7)
    conv7  = SeparableConv2D(256, 3, activation='relu', padding='same', **initializer)(conv7)
    conv7  = BatchNormalization()(conv7)
    
    up8    = Conv2DTranspose(128, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    merge8 = concatenate([conv2, up8], axis = 3)
    conv8  = SeparableConv2D(128, 3, activation='relu', padding='same', **initializer)(merge8)
    conv8  = BatchNormalization()(conv8)
    conv8  = SeparableConv2D(128, 3, activation='relu', padding='same', **initializer)(conv8)    
    conv8  = BatchNormalization()(conv8)
    
    up9    = Conv2DTranspose(64, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    merge9 = concatenate([conv1, up9], axis = 3)
    conv9  = SeparableConv2D(64, 3, activation='relu', padding='same', **initializer)(merge9)
    conv9  = BatchNormalization()(conv9)
    conv9  = SeparableConv2D(64, 3, activation='relu', padding='same', **initializer)(conv9)        
    conv9  = BatchNormalization()(conv9)
    conv9  = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    model = Model(input = inputs, output = conv10)
    model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

