# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:18:35 2020

@author: Nishidh Shekhawat
"""

from __future__ import print_function
import keras
from keras.datasets import cifar10
import tensorflow as tf
import os
import datetime 
from matplotlib import pyplot



class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

def set_hyperparameters():
    """    
    Sets the basic training parameters
    
    Returns
    -------
    batch_size - Hyperparameter that defines the number of samples to work through before updating the internal model parameters.
    epochs -  Hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.
    data_augmentation - Boolean, set True for enabling data augmentaion
    num_classes - Classes in the dataset, cifar10 has 10 classes

    """
    batch_size = 32 # 32
    epochs = 100 # 100
    data_augmentation = True
    num_classes = 10 
    
    return batch_size,epochs,data_augmentation,num_classes


def conv_net():
    """
    Defines the model 

    Returns
    -------
    Model with 14 layers

    """
    model = tf.keras.Sequential([
        
        # Layers 1 and 2 
        tf.keras.layers.Conv2D(64, (3, 3), padding='same',input_shape=(32, 32, 3)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        
        # Layers 3 and 4 
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'),
        tf.keras.layers.BatchNormalization(),
    
        # Layers 5 and 6  
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'),
        tf.keras.layers.BatchNormalization(),
    
        # Layers 7 and 8  
        tf.keras.layers.Conv2D(512, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'),
        tf.keras.layers.BatchNormalization(),
    
        # Layer 9 
        tf.keras.layers.Flatten(),
        
        # Layer 10 
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        
        # Layer 11 
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        
        # Layer 12  
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.25),
    
        # Layer 13 
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        
        # Layer 14
        tf.keras.layers.Dense(10)    
        ])
    
    return model

def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	pyplot.show()
	#pyplot.close()


if __name__ == "__main__":
    
    modelname= 'Layer 14'
    
    batch_size,epochs,data_augmentation,num_classes = set_hyperparameters()
    
    model = conv_net()
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.summary()
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    
    if not data_augmentation:
        print('Not using data augmentation.')
        A = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True
                  )
        summarize_diagnostics(A)
        
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            # Set input mean to 0 over the dataset, feature-wise
            featurewise_center=False, 
            # Set input mean to 0 over the dataset, feature-wise
            samplewise_center=False,  
            # Divide inputs by std of the dataset, feature-wise
            featurewise_std_normalization=False, 
            # Divide each input by its std
            samplewise_std_normalization=False, 
            # Apply ZCA whitening
            zca_whitening=False, 
            # Epsilon for ZCA whitening
            zca_epsilon=1e-06, 
            # Eandomly rotate images in the range (degrees, 0 to 180) 
            rotation_range=0, 
            # Randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # Randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            # Set range for random shear
            shear_range=0., 
            # Set range for random zoom
            zoom_range=0.,  
            # Set range for random channel shifts
            channel_shift_range=0.,  
            # Set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # Value used for fill_mode = "constant"
            cval=0.,  
            # Randomly flip images
            horizontal_flip=True,
            vertical_flip=False,  
            # Set rescaling factor (applied before any other transformation)
            rescale=None,
            # Set function that will be applied on each input
            preprocessing_function=None,
            # Image data format, either "channels_first" or "channels_last"
            data_format=None,
            # Fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        datagen.fit(x_train)
    
        # Fit the model on the batches generated by datagen.flow(). fit_generator
        A = model.fit(datagen.flow(x_train, y_train,
                                          batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            workers=4)
        summarize_diagnostics(A)
    
        
    # Save model and weights
    save_dir = os.path.join(os.getcwd(), 'Deeplearning-Models\saved_models')
    model_name = 'cifar10_trained_model_'+modelname+'_batch_'+str(batch_size)+'_epoch_'+str(epochs) +'_aug_'+str(data_augmentation)+'_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.h5'  
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
