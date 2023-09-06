#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import optimizers
from sklearn import preprocessing
import utils

with open("//home//bedek//Documents/ML//server do ml//config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

path = config['path_to_training_model_data']
path_to_save_model = config['path_to_save_trained_model']
name_of_model = config['name_of_saved_model']

def read_data(flatten=True):
    folders = os.listdir(path)
    folders.remove("report.txt")
    #folders.remove("rejected")

    for s in folders:
        path_to_file = path+"/"+s
    
        for filename in glob.glob(os.path.join(path_to_file, '*.csv')):
            database_df = pd.read_table(filename, delimiter=';', decimal=',', header=None)
            if flatten:
                X.append(utils.prep_data(database_df))
            else:
                X.append(utils.prep_data(database_df, flatten = False))
            Y.append(float(s))

    return X, Y

def train_model_flattened(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    X = preprocessing.normalize(X)

    print(X.shape)
    print(Y.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=.30,
        random_state=42
    )


    model.fit(x_train, y_train, epochs = 19)
    model.save(path_to_save_model + name_of_model)
    val_loss, val_acc = model.evaluate(x_test, y_test)
    tf.keras.backend.clear_session()

    print("Accuracy score for neural network: %.3f (%.3f)" % (val_acc, val_loss))

# najlepsze wyniki dla kanalow 
def train_model_per_channel(X, Y, channel):

    partial_model = Sequential()
    partial_model.add(Dense(1024, activation=tf.nn.relu))
    partial_model.add(Dense(512, activation=tf.nn.relu))
    partial_model.add(Dense(512, activation=tf.nn.relu))
    partial_model.add(Dense(256, activation=tf.nn.relu))
    partial_model.add(Dense(7, activation = tf.nn.softmax))

    partial_model.compile(optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

    X = np.array(X)
    Y = np.array(Y)
    X = preprocessing.normalize(X)

    print(X.shape)
    print(Y.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=.15,
        random_state=42
    )

    partial_model.fit(x_train, y_train, epochs = 10)
    partial_model.save(path_to_save_model, "_for_channel_", channel, name_of_model)
    val_loss, val_acc = partial_model.evaluate(x_test, y_test)
    tf.keras.backend.clear_session()

    print("Accuracy score for neural network: %.3f (%.3f)" % (val_acc, val_loss))

    return "Accuracy score for neural network for channel %.1f: %.3f (%.3f)" % (channel, val_acc, val_loss)

X=[]
Y=[]




model = Sequential()
model.add(Dense(1024, activation=tf.nn.relu))
model.add(Dense(512, activation=tf.nn.relu))
model.add(Dense(512, activation=tf.nn.relu))
model.add(Dense(256, activation=tf.nn.relu))
model.add(Dense(7, activation = tf.nn.softmax))

model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

X, Y = read_data()
train_model_flattened(X, Y)
# X, Y = read_data(flatten=False)
# X = np.array(X)
# Y = np.array(Y)
# print(X.shape)
# print(Y.shape)

# accuracies_per_channel =[]
# for i in range(8):
#     accuracies_per_channel.append(train_model_per_channel(X[:,:,i], Y, i))

# for acc in accuracies_per_channel:
#     print(acc)
        
# X = np.array(X)
# Y = np.array(Y)
# X = preprocessing.normalize(X)

# print(X.shape)
# print(Y.shape)


# x_train, x_test, y_train, y_test = train_test_split(
#     X, Y,
#     test_size=.30,
#     random_state=42
# )


# model.fit(x_train, y_train, epochs = 1)
# model.save(path_to_save_model + name_of_model)
# val_loss, val_acc = model.evaluate(x_test, y_test)
# tf.keras.backend.clear_session()

# print("Accuracy score for neural network: %.3f (%.3f)" % (val_acc, val_loss))

