import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import tensorflow as tf
import yaml
import gui
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
from sklearn import preprocessing
from threading import Thread


def prep_data(database, flatten = True):
    X = []
    database = delete_unwanted_columns(database)
    database = calulate_fft(database)
    if flatten:
        database = database.flatten()
    database= np.array(database)
    X = database
    
    return(X)

# deletes every other column and columns containing only 0
def delete_unwanted_columns(database):
    channel_headers_placeholder = []
    channel_to_delete_placefolder = []

    for i in range(64):
        channel_headers_placeholder.append("channel_"+str(i+1))
        
    for i in range(1, 33):
        channel_to_delete_placefolder.append("channel_"+str(i*2))
    

    database_df = pd.DataFrame(database)
    database_df.columns = channel_headers_placeholder

    database_df.drop(channel_to_delete_placefolder, axis=1, inplace=True)
    database_df = database_df.loc[:, (database_df != 0).any(axis=0)]
    
    #plt.plot(database_df["channel_1"])
    #plt.show()
    return(database_df)

def calulate_fft(database):
    database_fft = np.fft.rfft(database, axis = 0)
    database_fft = np.abs(database_fft)

    return database_fft

def nn_classification(database, label):
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    path_to_load_model = config['path_to_read_trained_model']
    X = np.empty([1,8008])
    X[0, :] = prep_data(database)
    X = preprocessing.normalize(X)
    

    model = keras.models.load_model(path_to_load_model)

    predicted_label = model.predict(X)

    tf.keras.backend.clear_session()

    return predicted_label


def prep_model_from_scratch(database, label):
    model = Sequential()
    model.add(Dense(1024, activation=tf.nn.relu))
    model.add(Dense(512, activation=tf.nn.relu))
    model.add(Dense(512, activation=tf.nn.relu))
    model.add(Dense(256, activation=tf.nn.relu))
    model.add(Dense(6, activation = tf.nn.softmax))

    model.compile(optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

## odpalic gui w glownym watku a flaska w dodatkowym
    