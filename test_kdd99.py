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

# In[46]:


columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label']

df = pd.read_csv(test_path,header=None,names=columns)

df = df[df['service']=='http']


df.describe()


features = df.columns.tolist()
features.remove('label')
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
        label = np.float(tmp_df[tmp_df['label']=='normal.'].shape[0]!=tmp_df.shape[0])
        # label = np.float(df.iloc[i+window_size]['label']!='normal.')

        yield x,label
    
# In[49]:


X = []
y = []
for x,label in get_windows(df,10,10):
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

