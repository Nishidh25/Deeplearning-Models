# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:18:06 2020

@author: Nishidh Shekhawat
"""
# https://missinglink.ai/guides/tensorflow/tensorflow-conv2d-layers-practical-guide/

import pickle
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import keras
#from keras.models import Model
import os

valid_features, valid_labels = pickle.load(open('preprocessed_validation.p', mode='rb'))


def conv2d(x,W,b,strides = 1):
    # common stride [1, 1, 1, 1] and [1, 2, 2, 1]  
    # Conv2D wrapper, with bias and relu activation using tensorflow
    x = tf.nn.conv2d(x, W,strides = [1, strides , strides , 1], padding = 'SAME')
    # Remove bias
    #x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x, k = 2):
    # MaxPool2D wrapper using tensorflow
    return tf.nn.max_pool(x , ksize=[1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')

def batch_norm(x):
    # Batch Normalizaton wrapper keras
    #return keras.layers.batch_normalization(x)
    #return tf.keras.batch_normalization(x)
    return tf.keras.layers.BatchNormalization()(x)

def fully_conn(x, outputs, keep_prob):
    # Fully connected layer wrapper with dropout and batch normailizaton
    x = tf.contrib.layers.fully_connected(inputs = x,num_outputs = outputs, activation_fn = tf.nn.relu)
    x = tf.nn.dropout(x,keep_prob) 
    return batch_norm(x)
    
def conv_net(X,weights,biases,keep_prob):
    # Data input X is a 4-D vector of 3,072 features per-image (10000*32*32*3 pixels per batch)
    
    
    # 14 layers total
    
    # 1 Conv layer 1 
    # conv1 = tf.nn.conv2d(X, [3, 3, 3, 64], strides=[1,1,1,1], padding='SAME')
    # conv1 = tf.nn.relu(conv1)
    conv1 = conv2d(X, weights['wc1'], biases['bc1'])
    
    # 2 Max Pool and batch norm 
    conv1_max = maxpool2d(conv1, k=2)
    #conv1_pool = tf.nn.max_pool(conv1, ksize = [1 ,2 , 2, 1] , strides = [1 ,2 ,2 , 1], padding = 'SAME')
    #conv1_norm = keras.layers.batch_normalization(conv1_pool)
    conv1_pool = batch_norm(conv1_max)    

    # 3 Conv layer 2 
    conv2 = conv2d(conv1_pool, weights['wc2'], biases['bc2'])
    # conv2 = tf.nn.conv2d(conv1_norm, [3, 3, 64, 128], strides=[1, 1 , 1, 1], padding='SAME')
    # conv2 = tf.nn.relu(conv2)
    
    # 4 Max Pool and batch norm
    conv2_max = maxpool2d(conv2, k=2)
    #conv2_pool = tf.nn.max_pool(conv2 , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #conv2_norm = keras.layers.batch_normalization(conv2_pool)
    conv2_pool = batch_norm(conv2_max)
    
    # 5 Conv layer 3 
    conv3 = conv2d(conv2_pool, weights['wc3'], biases['bc3'])
    
    # 6 Max Pool and batch norm  
    conv3_max = maxpool2d(conv3, k=2)
    conv3_pool = batch_norm(conv3_max)

    # 7 Conv layer 4 
    conv4 = conv2d(conv3_pool, weights['wc4'], biases['bc4'])
    
    # 8 Max Pool and batch norm  
    conv4_max = maxpool2d(conv4, k=2)
    conv4_pool = batch_norm(conv4_max)
    
    # 9 Flattening the 3-D output of the last convolutional operations
    #flat = tf.layers.flatten(conv4_pool)
    flat = tf.keras.layers.Flatten()(conv4_pool)
    
    # 10 Fully connected layer with 128 units
    full1 = fully_conn(flat, 128, keep_prob)
    
    # 11 Fully connected layer with 256 units
    full2 = fully_conn(full1, 256, keep_prob)
    
    # 12 Fully connected layer with 512 units
    full3 = fully_conn(full2, 512, keep_prob)
    
    # 13 Fully connected layer with 1024 units
    full4 = fully_conn(full3, 1024, keep_prob)
    
    # 14 Fully connected layer with 10 units 
    out = tf.contrib.layers.fully_connected(inputs = full4 ,num_outputs = 10 ,activation_fn = None)
    
    return out


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer, 
                feed_dict={
                    x: feature_batch,
                    y: label_batch,
                    keep_prob: keep_probability
                })


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    loss = session.run(cost, 
                    feed_dict={
                        x: feature_batch,
                        y: label_batch,
                        keep_prob: 1.
                    })
    
    
    
    
    valid_acc = session.run(accuracy, 
                         feed_dict={
                             x: valid_features,
                             y: valid_labels,
                             keep_prob: 1.
                         })
    
    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocessed_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


if __name__ == "__main__" :
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf1.disable_v2_behavior() 
    tf1.reset_default_graph()
    # inputs for model
    x = tf1.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_X')
    y = tf1.placeholder(tf.float32, shape=(None, 10), name='output_Y')   
    # keep_prob for dropout 
    keep_prob = tf1.placeholder(tf.float32, name='keep_prob')
    print(tf.__version__)
    #conv1_filter = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
    #print(conv1_filter)
    
    # Hyperparameters / Learning Parameters
    epochs = 10
    batch_size = 128
    
    #1.0 means no dropout, and 0.0 means no outputs from the layer.
    #A good value for dropout in a hidden layer is between 0.5 and 0.8. Input layers use a larger dropout rate, such as of 0.8.
    keep_probability = 0.7
    learning_rate = 0.001


    
    
    # Store layers weight & bias we can also use normal - tf.random_normal 
    # https://stackoverflow.com/questions/41704484/what-is-difference-between-tf-truncated-normal-and-tf-random-normal
    weights = {
        # 3x3 conv, 1 input, 64 outputs
        'wc1': tf.Variable(tf.random.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08)),
        # 3x3 conv, 64 inputs, 128 outputs
        'wc2': tf.Variable(tf.random.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08)),
        # 5x5 conv, 128 inputs, 256 outputs
        'wc3': tf.Variable(tf.random.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08)),
        # 5x5 conv, 256 inputs, 512 outputs
        'wc4': tf.Variable(tf.random.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08)),
    
        # fully connected, 7*7*64 inputs, 1024 outputs
        # 1024 inputs, 10 outputs (class prediction)
        # 'out': tf.Variable(tf.random_normal([1024, num_classes]))
        }



    biases = {
        'bc1': tf.Variable(tf.random.truncated_normal([64])),
        'bc2': tf.Variable(tf.random.truncated_normal([128])),
        'bc3': tf.Variable(tf.random.truncated_normal([256])),
        'bc4': tf.Variable(tf.random.truncated_normal([512])),
        #'out': tf.Variable(tf.random_normal([num_classes]))
        }
    
    print(x.shape)
    logits = conv_net(x,weights,biases,keep_prob)
    model = tf.identity(logits, name='logits')
    
    # Optional softmax we don't need it mostly 
    # prediction = tf.nn.softmax(logits)
    
    # Loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Evaluate model by checking accuracy
    # Use tf.argmax(prediction, 1) when using softmax
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32) , name = 'accuracy')
    
    save_model_path = './model_cifar-10_image_classification'
    print('Training...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())
        # Training cycle
        for epoch in range(epochs):
        # Loop over all batches
            n_batches = 5
            for batch_i in range(1, n_batches + 1):
                for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                    train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                
                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                print_stats(sess, batch_features, batch_labels, cost, accuracy)
    
    
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
    