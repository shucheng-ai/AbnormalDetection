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


columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label']

if use_all:
    df = pd.read_csv(full_path,header=None,names=columns)

else:
    df = pd.read_csv(ten_percent_path,header=None,names=columns)

df = df[df['service']=='http']

df = df[df['label']=='normal.']


features = df.columns.tolist()
features.remove('label')
new_features = []
for f in features:
    if df[f].dtype!='object':
        new_features.append(f)
features = new_features
print(len(new_features))



scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
pd.to_pickle(scaler,scaler_path)

def get_windows(df,window_size=10,stride=5):
    
    for i in tqdm(range(0,df.shape[0]-window_size+1,stride)):
        x = df.iloc[i:i+window_size][features].values
        label = np.int(df.iloc[i+window_size]['label']=='normal')
        yield x,label




X = []
y = []
for x,label in get_windows(df,10,5):
    X.append(x)
    y.append(label)


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
