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

def train_model(X, Y, channel):
    partial_model = Sequential()
    partial_model.add(Dense(1024, activation=tf.nn.relu))
    partial_model.add(Dense(512, activation=tf.nn.relu))
    partial_model.add(Dense(512, activation=tf.nn.relu))
    partial_model.add(Dense(256, activation=tf.nn.relu))
    partial_model.add(Dense(len(folders)+1, activation = tf.nn.softmax))

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

X=[]
Y=[]

with open("//home//bedek//Documents/ML//server do ml//config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

path = config['path_to_training_model_data']
path_to_save_model = config['path_to_save_trained_model']
name_of_model = config['name_of_saved_model']

# path = r"C:\Users\macie\Documents\ML\praca inzynierska\testy"

channel_headers_placeholder = []
channel_to_delete_placefolder = []

for i in range(64):
    channel_headers_placeholder.append("channel_"+str(i+1))

for i in range(1, 33):
    channel_to_delete_placefolder.append("channel_"+str(i*2))
    
folders = os.listdir(path)
folders.remove("report.txt")
folders.remove("rejected")

model = Sequential()
model.add(Dense(1024, activation=tf.nn.relu))
model.add(Dense(512, activation=tf.nn.relu))
model.add(Dense(512, activation=tf.nn.relu))
model.add(Dense(256, activation=tf.nn.relu))
model.add(Dense(len(folders)+1, activation = tf.nn.softmax))

model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

for s in folders:
    path_to_file = path+"/"+s
    
    for filename in glob.glob(os.path.join(path_to_file, '*.csv')):
        database_df = pd.read_table(filename, delimiter=';', decimal=',', names=channel_headers_placeholder)
        
        database_df.drop(channel_to_delete_placefolder, axis=1, inplace=True)
        database_df = database_df.loc[:, (database_df != 0).any(axis=0)]
        database_fft = np.fft.rfft(database_df, axis = 0)
        database_fft = np.abs(database_fft)
        
        # print("shape after fft: ", database_fft.shape)

        # database_fft = database_fft.flatten()

        
        
        database_fft = database_fft.tolist()
        X.append(database_fft)
        Y.append(float(s))
X = np.array(X)
Y = np.array(Y)
for i in range(8):
    train_model(X[:,:,i], Y, i)
        
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

