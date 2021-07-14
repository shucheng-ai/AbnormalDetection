import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.autograd import grad

import torch
import numpy as np
from tqdm import tqdm
import time
from datetime import timedelta
from models import *
from config import *



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# 评估函数，返回loss，重构loss，kld loss
def evaluate( model, data_iter, test=False,kld_weight=0.005):

    model.eval()
    loss_total = 0
    recon_loss_total = 0
    kld_loss_total = 0

    with torch.no_grad():
        for trains,labels in data_iter:
            labels = labels.view(-1)
            
            
            outputs, mu, log_var = model(trains)

            loss,recon_loss,kld_loss = model.loss_function(outputs,trains,mu,log_var,kld_weight=kld_weight)
            
            loss_total += loss.item()
            recon_loss_total += recon_loss.item()
            kld_loss_total += kld_loss.item()
            
            
    return loss_total / len(data_iter),recon_loss_total / len(data_iter),kld_loss_total / len(data_iter)



# 预测函数，返回原始y和预测标签y
def predict( model, data_iter, test=False,attack=True):


    model.eval()
    loss_total = 0
    recon_loss_total = 0
    kld_loss_total = 0

    predict_all = np.array([], dtype=float)

    labels_all = np.array([], dtype=int)

    # 初始化FGM
    fgm = FGM(model)
    # with torch.no_grad():
    for trains,labels in data_iter:
        labels = labels.view(-1)

        outputs, mu, log_var = model(trains)
        
        # # 计算重构loss
        # recon_loss = torch.abs(outputs - trains) 
        # recon_loss = torch.sum(recon_loss,axis=1)
        # recon_loss = torch.sum(recon_loss,axis=1)
        
        loss,_,_ = model.loss_function(outputs,trains,mu,log_var,kld_weight=kld_weight)
        loss.backward()
        
        # 是否对抗训练模型
        if attack:        
            fgm.attack(emb_name=emb_name,epsilon=epsilon)
            outputs, mu, log_var = model(trains)
            fgm.restore(emb_name=emb_name)
        model.zero_grad()
        
        # # 计算重构loss
        # recon_loss = torch.abs(outputs - trains) 
        # recon_loss = torch.sum(recon_loss,axis=1)
        # recon_loss = torch.sum(recon_loss,axis=1)
        
        # 计算重构loss
        bs,len_seq,channel_size = trains.size()
        recon_loss = torch.mean((outputs.reshape(bs,-1) - trains.reshape(bs,-1))**2,axis=1)
        # recon_loss = torch.sum(recon_loss,axis=1)
        # recon_loss = torch.sum(recon_loss,axis=1)
        
        labels = labels.to('cpu').numpy()
        predic = recon_loss.detach().cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)
    
    return labels_all,predict_all
