import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import tensorflow as tf
import yaml

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
    X = np.empty([1,8008])
    database = delete_unwanted_columns(database)
    database = calulate_fft(database)
    database = database.flatten()
    database= np.array(database)
    X[0,:] = database
    path_to_load_model = config['path_to_read_trained_model']

    print(X.shape)

    model = keras.models.load_model(path_to_load_model)

    predicted_label = model.predict(X)

    tf.keras.backend.clear_session()

    print("Predicted label: " + str(predicted_label) + " Actual label: " + str(label))