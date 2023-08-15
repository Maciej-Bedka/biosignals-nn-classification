#!/usr/bin/env python
# coding: utf-8

# Pobieranie danych z serwera

# In[1]:


import requests
import numpy as np
import matplotlib.pyplot as plt

raw = requests.get('http://127.0.0.1:5000/data')
raw_json = raw.json()
label = raw_json.get('calss_val')
database_raw = raw_json.get('data')


# In[2]:


print(type(database_raw))


# Tworzenie pandas dataframe na podstawie danych

# In[3]:


import pandas as pd
channel_headers_placeholder = []
channel_to_delete_placefolder = []

for i in range(64):
    channel_headers_placeholder.append("channel_"+str(i+1))
    
for i in range(1, 33):
    channel_to_delete_placefolder.append("channel_"+str(i*2))
                                       
database_df = pd.DataFrame(database_raw)
database_df.columns = channel_headers_placeholder

database_df.drop(channel_to_delete_placefolder, axis=1, inplace=True)

print(database_df)


# Otrzymany sygnal po usunieciu pustych kanalow i kanalow glosnikow

# In[4]:


database_df = database_df.loc[:, (database_df != 0).any(axis=0)]
plt.plot(database_df["channel_1"])
plt.show()
print(database_df)


# Przeprowadzanie szybkiej transformaty Fouriera

# In[11]:


database_fft = np.fft.rfft(database_df, axis = 0)
database_fft = np.abs(database_fft)

plt.plot(database_fft[:, 2])
plt.show()

print(database_fft.shape[1])


# model nn

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf

model = Sequential()
model.add(Dense(16, activation=tf.nn.relu))
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dense(6, activation = tf.nn.softmax))

