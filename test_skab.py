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
import glob
# In[46]:



columns = [
'Accelerometer1RMS',
'Accelerometer2RMS',
'Current','Pressure',
'Temperature','Thermocouple','Voltage','Volume Flow RateRMS',
'anomaly'
]



X = []
y = []

for p in glob.glob('SKAB-master\\data\\valve*\\*.csv'):
    print(p)
    df = pd.read_csv(p,sep=';')[columns].fillna(-1)


    df.describe()


    features = df.columns.tolist()
    features.remove('anomaly')
    new_features = []
    for f in features:
        if df[f].dtype!='object':
            new_features.append(f)
    features = new_features
    print(features)
    print(len(new_features))



    scaler = pd.read_pickle(scaler_path)
    df[features] = scaler.transform(df[features])


    from tqdm import tqdm as tqdm
    def get_windows(df,window_size=10,stride=5):
        
        for i in tqdm(range(0,df.shape[0]-window_size+1,stride)):

            tmp_df = df.iloc[i:i+window_size]
            x = tmp_df[features].values
            label = np.float(tmp_df[tmp_df['anomaly']==0].shape[0]!=tmp_df.shape[0])
            # label = np.float(df.iloc[i+window_size]['label']!='normal.')

            yield x,label
        
    # In[49]:

    for x,label in get_windows(df,window_size,1):
        X.append(x)
        y.append(label)


    # In[50]:


X =np.array(X)
y = np.array(y)


print(y.mean())
print(y)


with h5py.File(test_data_path, 'w') as hf:
    hf['X'] = X
    hf['y'] = y

