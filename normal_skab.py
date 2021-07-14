#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import TimeSeriesSplit
import h5py
import os
from config import *


columns = [
'Accelerometer1RMS',
'Accelerometer2RMS',
'Current','Pressure',
'Temperature','Thermocouple','Voltage','Volume Flow RateRMS',
]


df = pd.read_csv(full_path,sep=';')[columns].fillna(-1)


features = df.columns.tolist()
new_features = []
for f in features:
    if df[f].dtype!='object':
        new_features.append(f)
features = new_features
print(features)
print(len(new_features))

print(df)


scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
pd.to_pickle(scaler,scaler_path)

def get_windows(df,window_size=10,stride=5):
    
    for i in tqdm(range(0,df.shape[0]-window_size+1,stride)):
        x = df.iloc[i:i+window_size][features].values
        anomaly = np.zeros(x.shape[0])
        yield x,anomaly



X = []
y = []
for x,anomaly in get_windows(df,window_size,1):
    X.append(x)
    y.append(anomaly)


X =np.array(X)
y = np.array(y)


tscv = TimeSeriesSplit()
for train_index, test_index in tscv.split(X):
    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]

X_train.shape,X_val.shape,y_train.shape,y_val.shape 


with h5py.File(train_data_path, 'w') as hf:
    hf['X_train'] = X_train
    hf['X_val'] = X_val
    hf['y_train'] = y_train
    hf['y_val'] = y_val
