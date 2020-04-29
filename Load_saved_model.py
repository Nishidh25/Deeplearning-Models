# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:09:13 2020

@author: Nishidh Shekhawat
"""
import tensorflow as tf
from keras.datasets import cifar10
import keras

# Loading model
model = tf.keras.models.load_model('C:\\Users\\Nishidh Shekhawat\\Deeplearning-Models\\saved_models\\cifar10_trained_model_Layer 14_batch_32_epoch_100_aug_True_20200429-200356.h5')

# Model Summary
model.summary()

# Loading Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
num_classes = 10

y_test = keras.utils.to_categorical(y_test, num_classes)
x_test = x_test.astype('float32')
x_test /= 255
    
# Evaluatuing the model with test sets
score = model.evaluate(x_test, y_test, verbose=1)