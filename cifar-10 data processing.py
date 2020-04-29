# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:51:45 2020

@author: Nishidh Shekhawat
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical


"""
https://www.cs.toronto.edu/~kriz/cifar.html   

Cifar-10 dataset download 
"""

def get_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_dataset_batch(path,batch_id):
    with open(path + '/data_batch_' + str(batch_id), mode ='rb') as file:
        
        # Encoding of cifar-10 batch is latin1 in this case not binary 
        batch = pickle.load(file,encoding = 'latin1') 
        
        # original data - batch['data'] is a 10000 x 3072 numpy array
        # We convert it into 10000 x 3 x 32 x 32 by reshaping it 
        # 10000 is number of images/samples, 3 is number of chanels , 32 is width , 32 is height
        # Transpose as tensorflow and matplotlib take (width, height, num_channel) and we have (num_channel, width, height)
        # Data will be reshaped and transposed 
        data = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)   
        labels = batch['labels'] 
    
    return data , labels


def load_test_set(path):
    with open(path + '/test_batch', mode='rb') as file:
        
        batch = pickle.load(file,encoding = 'latin1') 
        
        # original data - batch['data'] is a 10000 x 3072 numpy array
        # We convert it into 10000 x 3 x 32 x 32 by reshaping it 
        # 10000 is number of images/samples, 3 is number of chanels , 32 is width , 32 is height
        # Transpose as tensorflow and matplotlib take (width, height, num_channel) and we have (num_channel, width, height)
        # Data will be reshaped and transposed 
        data = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)   
        labels = batch['labels'] 
    
    return data , labels

    

def visualize_data(data,labels,numb):
    
    
    image = data[numb]
    label = labels[numb]
    
    print("Image : ", numb)
    print("Shape", image.shape)
    print("Label = ",  label , ", Name = " , get_label_names()[label])
    plt.imshow(image)

def normalize(X):
    # Max - Min normalization 
    X_min = np.min(X)
    X_max = np.max(X)
    X = (X - X_min)/(X_max-X_min)
    return X

def one_hot_encode(X):
    # One Hot Encode with Keras  https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    X = np.array(X)
    # one hot encode
    encoded = to_categorical(X)
    
    return encoded

def preprocess_and_save(data, labels, filename):
    data = normalize(data)
    labels = one_hot_encode(labels)

    pickle.dump((data, labels), open(filename, 'wb'))



def load_process_and_save_data(path):
    batches = 5
    # below lists will be used for saving validation data
    valid_data = [] 
    valid_labels = []
    
    for batch in range(1,batches+1):
        print("Preprocessing Batch : ", batch)
        data , labels = load_dataset_batch(path,batch)
        
        index_of_validation = int(len(data) * 0.1) # 10% is validation data
        
        # saving the starting 90% train data first 
        preprocess_and_save(data[:-index_of_validation], labels[:-index_of_validation], 'preprocessed_batch_' + str(batch) + '.p')
        
        # adding 10% validation data to list
        valid_data.extend(data[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])
        print("Done Preprocessing Batch : ", batch)
        
    
    # saving the ending 10% validation data
    print("Preprocessing Validation Set... ")
    preprocess_and_save(np.array(valid_data), np.array(valid_labels),'preprocessed_validation.p')
    print("Done Preprocessing Validation Set... ")
    
    # for test
    print("Preprocessing Test Set... ")
    test_data , test_labels = load_test_set(path)
    preprocess_and_save(np.array(test_data), np.array(test_labels),'preprocessed_training.p')
    print("Done Preprocessing Test Set... ")     


def check_preprocessed():
    

    pass

if __name__ == "__main__":
    path = "C:/cifar10-python/cifar-10-batches-py"
    # data , labels = load_dataset_batch(path,3)
    # #print(data)
    # visualize_data(data,labels,100)
    
    # data = normalize(data)
    # labels = one_hot_encode(labels)
    
    load_process_and_save_data(path)





    # print(data[1])
    # print(labels[1])
    
    
    #https://medium.com/@joeyism/creating-alexnet-on-tensorflow-from-scratch-part-1-getting-cifar-10-data-46d349a4282f
    #https://medium.com/@joeyism/creating-alexnet-on-tensorflow-from-scratch-part-2-creating-alexnet-e0cd948d7b04
    #https://github.com/deep-diver/CIFAR10-img-classification-tensorflow/blob/master/CIFAR10_image_classification.ipynb
    #https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c
    
