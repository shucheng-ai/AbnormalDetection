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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
device = torch.device('cuda') 
from torch.nn.utils import weight_norm



## attention layer
class SimpleAttention(nn.Module):
    def __init__(self,input_size):
        super(SimpleAttention,self).__init__()
        self.input_size = input_size
        self.word_weight = nn.Parameter(torch.Tensor(self.input_size))
        self.word_bias = nn.Parameter(torch.Tensor(1))
        self._create_weights()

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.word_bias.data.normal_(mean, std)


    def forward(self,inputs):
        
        att = torch.einsum('abc,c->ab',(inputs,self.word_weight))
        att = att+self.word_bias
        att = torch.tanh(att)

        att = torch.exp(att)
        s = torch.sum(att,1,keepdim=True)+1e-6
        att = att / s

        att = torch.einsum('abc,ab->ac',(inputs,att))

        return att

# 对抗训练Fast Gradient Method
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.9, emb_name=["W",'position_enc','cat_enc']):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and any([p in name for p in emb_name]):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name=["W",'position_enc','cat_enc']):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and any([p in name for p in emb_name]): 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}        

# Beta VBAE LSTM
class AE_LSTM(nn.Module):
    num_iter = 0
    def __init__(self,input_dim,hidden_dim,max_len=10,loss_type='B',
                 max_capacity = 25,
                 Capacity_max_iter = 10000,
                 beta = 4,
                 gamma = 10.,
        ):

        super(AE_LSTM,self).__init__()
        self.loss_type = loss_type
        self.max_len = max_len # 输入序列最大长度
        self.C_max = torch.Tensor([max_capacity]).to(device)
        self.C_stop_iter = Capacity_max_iter
        self.beta = beta 
        self.gamma = gamma

        # 编码器
        self.gru_encoder = nn.GRU(input_size=input_dim,hidden_size=hidden_dim,batch_first=True,bidirectional=True,num_layers=1)

        # 解码器
        self.gru_decoder = nn.GRU(input_size=hidden_dim*2,hidden_size=hidden_dim,batch_first=True,bidirectional=True,num_layers=1)

        # mu层
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            )

        # log_var层
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            )


        self.decoder_input = nn.Linear(hidden_dim, hidden_dim*2*max_len)


        self.decoder_transform_weight = nn.Parameter(torch.Tensor(hidden_dim,max_len,hidden_dim*2))
        self._create_weights()
        self.out = nn.Linear(hidden_dim*2, input_dim)

    # 创建权重
    def _create_weights(self, mean=0.0, std=0.05):
        self.decoder_transform_weight.data.normal_(mean, std)


    # 重参数化
    def reparameterize(self, mu, log_var):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param log_var: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
        
    # 采样生成新样本
    def sample(self,num_samples):
        
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        z = torch.randn(num_samples,
                        self.hidden_dim)

        z = z.to(device)

        samples = self.decode(z)
        
        return samples


    # 编码器
    def encode(self, x):
        bs,len_seq,channel_size = x.size()
        x,_ = self.gru_encoder(x)

        x = torch.mean(x,1)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    # 解码器
    def decode(self,x):
        bs,hidden_dim = x.size()

        x = torch.einsum('bcd,ab->acd',[self.decoder_transform_weight,x])
        x = F.relu(x)

        x,_ = self.gru_decoder(x)

        outputs = self.out(x)
        return outputs
        
        
    # loss函数
    def loss_function(self,
                      output,label,mu,log_var,kld_weight=0.005):
        self.num_iter += 1
        recons = output
        input = label

        # 此处使用L1 loss 即MAE loss
        recons_loss = nn.L1Loss()(recons, input)
        # 计算kld loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # 合并，具体两种类型可参照论文
        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss  + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return loss,recons_loss, kld_loss


    # 串联执行，返回结果
    def forward(self, x):

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]


    # 直接使用std和mean重建样本
    def reconstruct(self, x):

        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        z = std + mu
        return  [self.decode(z), mu, log_var]



# Beta VBAE ATT_LSTM
class AE_ATT_LSTM(nn.Module):
    num_iter = 0
    def __init__(self,input_dim,hidden_dim,max_len=10,loss_type='B',
                 max_capacity = 25,
                 Capacity_max_iter = 10000,
                 beta = 4,
                 gamma = 10.,
        ):

        super(AE_ATT_LSTM,self).__init__()
        self.loss_type = loss_type
        self.max_len = max_len # 输入序列最大长度
        self.C_max = torch.Tensor([max_capacity]).to(device)
        self.C_stop_iter = Capacity_max_iter
        self.beta = beta 
        self.gamma = gamma

        # 编码器
        self.gru_encoder = nn.GRU(input_size=input_dim,hidden_size=hidden_dim,batch_first=True,bidirectional=True,num_layers=2)

        # 解码器
        self.gru_decoder = nn.GRU(input_size=hidden_dim*2,hidden_size=hidden_dim,batch_first=True,bidirectional=True,num_layers=1)


        self.att = SimpleAttention(hidden_dim*2)

        # mu层
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            )

        # log_var层
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            )


        self.decoder_input = nn.Linear(hidden_dim, hidden_dim*2*max_len)


        self.decoder_transform_weight = nn.Parameter(torch.Tensor(hidden_dim,max_len,hidden_dim*2))
        self._create_weights()
        self.out = nn.Linear(hidden_dim*2, input_dim)

    # 创建权重
    def _create_weights(self, mean=0.0, std=0.05):
        self.decoder_transform_weight.data.normal_(mean, std)


    # 重参数化
    def reparameterize(self, mu, log_var):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param log_var: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
        
    # 采样生成新样本
    def sample(self,num_samples):
        
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        z = torch.randn(num_samples,
                        self.hidden_dim)

        z = z.to(device)

        samples = self.decode(z)
        
        return samples


    # 编码器
    def encode(self, x):
        bs,len_seq,channel_size = x.size()
        x,_ = self.gru_encoder(x)

        x = self.att(x)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    # 解码器
    def decode(self,x):
        bs,hidden_dim = x.size()

        x = torch.einsum('bcd,ab->acd',[self.decoder_transform_weight,x])
        x = F.relu(x)

        x,_ = self.gru_decoder(x)

        outputs = self.out(x)
        return outputs
        
        
    # loss函数
    def loss_function(self,
                      output,label,mu,log_var,kld_weight=0.005):
        self.num_iter += 1
        recons = output
        input = label

        # 此处使用L1 loss 即MAE loss
        recons_loss = nn.L1Loss()(recons, input)
        # 计算kld loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # 合并，具体两种类型可参照论文
        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss  + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return loss,recons_loss, kld_loss


    # 串联执行，返回结果
    def forward(self, x):

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]


    # 直接使用std和mean重建样本
    def reconstruct(self, x):

        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        z = std + mu
        return  [self.decode(z), mu, log_var]





class LightweightConv1d(nn.Module):
    '''Lightweight Convolution assuming the input is BxCxT
    This is just an example that explains LightConv clearer than the TBC version.
    We don't use this module in the model.
    Args:
        input_size: # of channels of the input and output
        kernel_size: convolution channels
        padding: padding
        num_heads: number of heads used. The weight is of shape
            `(num_heads, 1, kernel_size)`
        weight_softmax: normalize the weight with softmax before the convolution
    Shape:
        Input: BxCxT, i.e. (batch_size, input_size, timesteps)
        Output: BxCxT, i.e. (batch_size, input_size, timesteps)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias: the learnable bias of the module of shape `(input_size)`
    '''

    def __init__(self, input_size, kernel_size=1, padding=1, num_heads=1,
                 weight_softmax=True, bias=True, weight_dropout=0.,dilation = 1):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))
        self.dilation = dilation
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None
        self.weight_dropout = weight_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, input):
        '''
        input size: B x C x T
        output size: B x C x T
        '''
        B, C, T = input.size()
        H = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)


        weight = F.dropout(weight, self.weight_dropout, training=self.training)
        # Merge every C/H entries into the batch dimension (C = self.input_size)
        # B x C x T -> (B * C/H) x H x T
        # One can also expand the weight to C x 1 x K by a factor of C/H
        # and do not reshape the input instead, which is slow though
        input = input.view(-1, H, T)

        output = F.conv1d(input, weight, padding=self.padding, groups=self.num_heads,dilation=self.dilation)

        output = output.view(B, C, -1)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        return output




# Beta VBAE ATT_LightConv
class AE_ATT_LightConv(nn.Module):
    num_iter = 0
    def __init__(self,input_dim,hidden_dim,max_len=10,loss_type='B',
                 max_capacity = 25,
                 Capacity_max_iter = 10000,
                 beta = 4,
                 gamma = 10.,
                 num_heads = 32,
                 padding = 1,
                 dilation = 1,
                 recon_loss_function=nn.MSELoss(),
        ):

        super(AE_ATT_LightConv,self).__init__()
        self.loss_type = loss_type
        self.max_len = max_len # 输入序列最大长度
        self.C_max = torch.Tensor([max_capacity]).to(device)
        self.C_stop_iter = Capacity_max_iter
        self.beta = beta 
        self.gamma = gamma
        self.recon_loss_function = recon_loss_function

        self.translation = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            )

        # 编码器
        self.conv_encoder = nn.Sequential(
            LightweightConv1d(hidden_dim, kernel_size=3, padding=padding,dilation=dilation, num_heads=num_heads,weight_softmax=True, bias=True, weight_dropout=0.1),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=1,padding=0),
            nn.GLU(dim=1),
            LightweightConv1d(hidden_dim, kernel_size=3, padding=padding,dilation=dilation, num_heads=num_heads,weight_softmax=True, bias=True, weight_dropout=0.1),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=1,padding=0),
            nn.GLU(dim=1),
            LightweightConv1d(hidden_dim, kernel_size=3, padding=padding,dilation=dilation, num_heads=num_heads,weight_softmax=True, bias=True, weight_dropout=0.1),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*4, kernel_size=1,padding=0),
            nn.GLU(dim=1),
            )



        # 解码器
        self.conv_decoder = nn.Sequential(
            LightweightConv1d(hidden_dim, kernel_size=3, padding=padding,dilation=dilation, num_heads=num_heads,weight_softmax=True, bias=True, weight_dropout=0.1),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=1,padding=0),
            nn.GLU(dim=1),
            LightweightConv1d(hidden_dim, kernel_size=3, padding=padding,dilation=dilation, num_heads=num_heads,weight_softmax=True, bias=True, weight_dropout=0.1),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=1,padding=0),
            nn.GLU(dim=1),
            LightweightConv1d(hidden_dim, kernel_size=3, padding=padding,dilation=dilation, num_heads=num_heads,weight_softmax=True, bias=True, weight_dropout=0.1),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*4, kernel_size=1,padding=0),
            nn.GLU(dim=1),
            )

        self.att = SimpleAttention(hidden_dim*2)

        # mu层
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            )

        # log_var层
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            )


        self.decoder_transform_weight = nn.Parameter(torch.Tensor(hidden_dim,max_len,hidden_dim))
        self._create_weights()
        self.out = nn.Linear(hidden_dim*2, input_dim)

    # 创建权重
    def _create_weights(self, mean=0.0, std=0.05):
        self.decoder_transform_weight.data.normal_(mean, std)


    # 重参数化
    def reparameterize(self, mu, log_var):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param log_var: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
        
    # 采样生成新样本
    def sample(self,num_samples):
        
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        z = torch.randn(num_samples,
                        self.hidden_dim)

        z = z.to(device)

        samples = self.decode(z)
        
        return samples


    # 编码器
    def encode(self, x):
        bs,len_seq,channel_size = x.size()
        x = self.translation(x)
        
        x = x.permute(0,2,1).contiguous()
        x = self.conv_encoder(x)
        x = x.permute(0,2,1).contiguous()
        x = self.att(x)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    # 解码器
    def decode(self,x):
        bs,hidden_dim = x.size()

        x = torch.einsum('bcd,ab->acd',[self.decoder_transform_weight,x])
        x = F.relu(x)

        x = x.permute(0,2,1).contiguous()
        x = self.conv_decoder(x)
        x = x.permute(0,2,1).contiguous()
        outputs = self.out(x)
        return outputs
        
        
    # loss函数
    def loss_function(self,
                      output,label,mu,log_var,kld_weight=0.005):
        self.num_iter += 1
        recons = output
        input = label

        # 此处使用L1 loss 即MAE loss
        recons_loss = self.recon_loss_function(recons, input)
        # 计算kld loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
        # 合并，具体两种类型可参照论文
        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss  + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return loss,recons_loss, kld_loss


    # 串联执行，返回结果
    def forward(self, x):

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]


    # 直接使用std和mean重建样本
    def reconstruct(self, x):

        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        z = std + mu
        return  [self.decode(z), mu, log_var]



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  #  裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class LightTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2,num_heads=16):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(LightTemporalBlock, self).__init__()
        self.conv1 = nn.Sequential(
            LightweightConv1d(n_inputs, kernel_size=kernel_size, padding=padding,dilation=dilation, num_heads=num_heads,weight_softmax=True, bias=True, weight_dropout=dropout),
            nn.Conv1d(in_channels=n_inputs, out_channels=n_outputs*2, kernel_size=1,padding=0),
            nn.GLU(dim=1),
            )

        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Sequential(
            LightweightConv1d(n_outputs, kernel_size=kernel_size, padding=padding,dilation=dilation, num_heads=num_heads,weight_softmax=True, bias=True, weight_dropout=dropout),
            nn.Conv1d(in_channels=n_outputs, out_channels=n_outputs*2, kernel_size=1,padding=0),
            nn.GLU(dim=1),
            )

        self.chomp2 = Chomp1d(padding)  #  裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()


        self.net = nn.Sequential(self.conv1, self.chomp1,
                                 self.conv2, self.chomp2,)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        # self.conv1.weight.data.normal_(0, 0.01)
        # self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class BiLightTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2,num_heads=16):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(BiLightTemporalBlock, self).__init__()
        
        self.forward_ltcn = LightTemporalBlock(n_inputs, n_outputs, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                     padding=padding, dropout=dropout,num_heads=num_heads)
        self.backward_ltcn = LightTemporalBlock(n_inputs, n_outputs, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                     padding=padding, dropout=dropout,num_heads=num_heads)



    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        
        x_backward = torch.flip(x,[2])

        x = self.forward_ltcn(x)

        x_backward = self.backward_ltcn(x_backward)

        x_backward = torch.flip(x_backward,[2])

        out = torch.cat([x,x_backward],1)

        return out

# Beta VBAE ATT_TCN
class AE_ATT_TCN(nn.Module):
    num_iter = 0
    def __init__(self,input_dim,hidden_dim,max_len=10,loss_type='B',
                 max_capacity = 25,
                 Capacity_max_iter = 10000,
                 beta = 4,
                 gamma = 10.,
                 num_heads = 32,
                 padding = 1,
                 dilation = 1,
        ):
        
        super(AE_ATT_TCN,self).__init__()
        self.loss_type = loss_type
        self.max_len = max_len # 输入序列最大长度
        self.C_max = torch.Tensor([max_capacity]).to(device)
        self.C_stop_iter = Capacity_max_iter
        self.beta = beta 
        self.gamma = gamma
        
        self.translation = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            )
        
        # 编码器
        self.conv_encoder = nn.Sequential(
            TemporalBlock(hidden_dim, hidden_dim, 3, stride=1, dilation=1,
                                     padding=2, dropout=0.1),
            TemporalBlock(hidden_dim, hidden_dim, 3, stride=1, dilation=2,
                                     padding=4, dropout=0.1),
            TemporalBlock(hidden_dim, hidden_dim*2, 3, stride=1, dilation=4,
                                     padding=8, dropout=0.1),
            )


        # 解码器
        self.conv_decoder = nn.Sequential(
            TemporalBlock(hidden_dim, hidden_dim, 3, stride=1, dilation=1,
                                     padding=2, dropout=0.1),
            TemporalBlock(hidden_dim, hidden_dim, 3, stride=1, dilation=2,
                                     padding=4, dropout=0.1),
            TemporalBlock(hidden_dim, hidden_dim*2, 3, stride=1, dilation=4,
                                     padding=8, dropout=0.1),
            )

        self.att = SimpleAttention(hidden_dim*2)

        # mu层
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            )

        # log_var层
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            )


        self.decoder_transform_weight = nn.Parameter(torch.Tensor(hidden_dim,max_len,hidden_dim))
        self._create_weights()
        self.out = nn.Linear(hidden_dim*2, input_dim)

    # 创建权重
    def _create_weights(self, mean=0.0, std=0.05):
        self.decoder_transform_weight.data.normal_(mean, std)


    # 重参数化
    def reparameterize(self, mu, log_var):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param log_var: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
        
    # 采样生成新样本
    def sample(self,num_samples):
        
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        z = torch.randn(num_samples,
                        self.hidden_dim)

        z = z.to(device)

        samples = self.decode(z)
        
        return samples


    # 编码器
    def encode(self, x):
        bs,len_seq,channel_size = x.size()
        x = self.translation(x)
        
        x = x.permute(0,2,1).contiguous()
        x = self.conv_encoder(x)
        # print(x.size())

        x = x.permute(0,2,1).contiguous()
        x = self.att(x)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    # 解码器
    def decode(self,x):
        bs,hidden_dim = x.size()

        x = torch.einsum('bcd,ab->acd',[self.decoder_transform_weight,x])
        x = F.relu(x)

        x = x.permute(0,2,1).contiguous()
        x = self.conv_decoder(x)
        x = x.permute(0,2,1).contiguous()
        outputs = self.out(x)
        return outputs
        
        
    # loss函数
    def loss_function(self,
                      output,label,mu,log_var,kld_weight=0.005):
        self.num_iter += 1
        recons = output
        input = label

        # 此处使用L1 loss 即MAE loss
        recons_loss = nn.L1Loss()(recons, input)
        # 计算kld loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # 合并，具体两种类型可参照论文
        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss  + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return loss,recons_loss, kld_loss


    # 串联执行，返回结果
    def forward(self, x):

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]


    # 直接使用std和mean重建样本
    def reconstruct(self, x):

        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        z = std + mu
        return  [self.decode(z), mu, log_var]



# Beta VBAE ATT_TCN
class AE_ATT_LightTCN(nn.Module):
    num_iter = 0
    def __init__(self,input_dim,hidden_dim,max_len=10,loss_type='B',
                 max_capacity = 25,
                 Capacity_max_iter = 10000,
                 beta = 4,
                 gamma = 10.,
                 num_heads = 32,
                 padding = 1,
                 dilation = 1,
                 recon_loss_function = nn.MSELoss()
        ):

        super(AE_ATT_LightTCN,self).__init__()
        self.loss_type = loss_type
        self.max_len = max_len # 输入序列最大长度
        self.C_max = torch.Tensor([max_capacity]).to(device)
        self.C_stop_iter = Capacity_max_iter
        self.beta = beta 
        self.gamma = gamma
        self.recon_loss_function =  recon_loss_function
        self.translation = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            )

        # 编码器
        self.conv_encoder = nn.Sequential(
            LightTemporalBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, dilation=1,
                                     padding=2, dropout=0.1,num_heads=num_heads),
            LightTemporalBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, dilation=2,
                                     padding=4, dropout=0.1,num_heads=num_heads),
            LightTemporalBlock(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, dilation=4,
                                     padding=8, dropout=0.1,num_heads=num_heads),
            )


        # 解码器
        self.conv_decoder = nn.Sequential(
            LightTemporalBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, dilation=1,
                                     padding=2, dropout=0.1,num_heads=num_heads),
            LightTemporalBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, dilation=2,
                                     padding=4, dropout=0.1,num_heads=num_heads),
            LightTemporalBlock(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, dilation=4,
                                     padding=8, dropout=0.1,num_heads=num_heads),
            )

        self.att = SimpleAttention(hidden_dim*2)

        # mu层
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            )

        # log_var层
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            )


        self.decoder_transform_weight = nn.Parameter(torch.Tensor(hidden_dim,max_len,hidden_dim))
        

        self.decoder_transform_linear= nn.Sequential(
            nn.Linear(hidden_dim,max_len*hidden_dim),
            nn.ReLU()
            )
        
        self._create_weights()
        self.out = nn.Linear(hidden_dim*2, input_dim)

    # 创建权重
    def _create_weights(self, mean=0.0, std=0.05):
        self.decoder_transform_weight.data.normal_(mean, std)


    # 重参数化
    def reparameterize(self, mu, log_var):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param log_var: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
        
    # 采样生成新样本
    def sample(self,num_samples):
        
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        z = torch.randn(num_samples,
                        self.hidden_dim)

        z = z.to(device)

        samples = self.decode(z)
        
        return samples


    # 编码器
    def encode(self, x):
        bs,len_seq,channel_size = x.size()
        x = self.translation(x)
        
        x = x.permute(0,2,1).contiguous()
        x = self.conv_encoder(x)
        # print(x.size())

        x = x.permute(0,2,1).contiguous()
        x = self.att(x)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    # 解码器
    def decode(self,x):
        bs,hidden_dim = x.size()

        # x = torch.einsum('bcd,ab->acd',[self.decoder_transform_weight,x])
        # x = F.relu(x)
        
        x = self.decoder_transform_linear(x).view(bs,self.max_len,hidden_dim)

        x = x.permute(0,2,1).contiguous()
        x = self.conv_decoder(x)
        x = x.permute(0,2,1).contiguous()
        outputs = self.out(x)
        return outputs
        
        
    # loss函数
    def loss_function(self,
                      output,label,mu,log_var,kld_weight=0.005):
        self.num_iter += 1
        recons = output
        input = label

        # 此处使用L1 loss 即MAE loss
        recons_loss = self.recon_loss_function(recons, input)
        # 计算kld loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # 合并，具体两种类型可参照论文
        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss  + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return loss,recons_loss, kld_loss


    # 串联执行，返回结果
    def forward(self, x):

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]


    # 直接使用std和mean重建样本
    def reconstruct(self, x):

        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        z = std + mu
        return  [self.decode(z), mu, log_var]




# Beta VBAE ATT_TCN
class AE_ATT_BiLightTCN(nn.Module):
    num_iter = 0
    def __init__(self,input_dim,hidden_dim,max_len=10,loss_type='B',
                 max_capacity = 25,
                 Capacity_max_iter = 10000,
                 beta = 4,
                 gamma = 10.,
                 num_heads = 32,
                 padding = 1,
                 dilation = 1,
        ):

        super(AE_ATT_BiLightTCN,self).__init__()
        self.loss_type = loss_type
        self.max_len = max_len # 输入序列最大长度
        self.C_max = torch.Tensor([max_capacity]).to(device)
        self.C_stop_iter = Capacity_max_iter
        self.beta = beta 
        self.gamma = gamma

        self.translation = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            )

        # 编码器
        self.conv_encoder = nn.Sequential(
            BiLightTemporalBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, dilation=1,
                                     padding=2, dropout=0.1,num_heads=num_heads),
            BiLightTemporalBlock(hidden_dim*2, hidden_dim, kernel_size=3, stride=1, dilation=2,
                                     padding=4, dropout=0.1,num_heads=num_heads),
            BiLightTemporalBlock(hidden_dim*2, hidden_dim, kernel_size=3, stride=1, dilation=4,
                                     padding=8, dropout=0.1,num_heads=num_heads),
            )


        # 解码器
        self.conv_decoder = nn.Sequential(
            BiLightTemporalBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, dilation=1,
                                     padding=2, dropout=0.1,num_heads=num_heads),
            BiLightTemporalBlock(hidden_dim*2, hidden_dim, kernel_size=3, stride=1, dilation=2,
                                     padding=4, dropout=0.1,num_heads=num_heads),
            BiLightTemporalBlock(hidden_dim*2, hidden_dim, kernel_size=3, stride=1, dilation=4,
                                     padding=8, dropout=0.1,num_heads=num_heads),
            )

        self.att = SimpleAttention(hidden_dim*2)

        # mu层
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            )

        # log_var层
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            )


        self.decoder_transform_weight = nn.Parameter(torch.Tensor(hidden_dim,max_len,hidden_dim))
        self._create_weights()
        self.out = nn.Linear(hidden_dim*2, input_dim)

    # 创建权重
    def _create_weights(self, mean=0.0, std=0.05):
        self.decoder_transform_weight.data.normal_(mean, std)


    # 重参数化
    def reparameterize(self, mu, log_var):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param log_var: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
        
    # 采样生成新样本
    def sample(self,num_samples):
        
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        z = torch.randn(num_samples,
                        self.hidden_dim)

        z = z.to(device)

        samples = self.decode(z)
        
        return samples


    # 编码器
    def encode(self, x):
        bs,len_seq,channel_size = x.size()
        x = self.translation(x)
        
        x = x.permute(0,2,1).contiguous()
        x = self.conv_encoder(x)
        # print(x.size())

        x = x.permute(0,2,1).contiguous()
        x = self.att(x)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    # 解码器
    def decode(self,x):
        bs,hidden_dim = x.size()

        x = torch.einsum('bcd,ab->acd',[self.decoder_transform_weight,x])
        x = F.relu(x)

        x = x.permute(0,2,1).contiguous()
        x = self.conv_decoder(x)
        x = x.permute(0,2,1).contiguous()
        outputs = self.out(x)
        return outputs
        
        
    # loss函数
    def loss_function(self,
                      output,label,mu,log_var,kld_weight=0.005):
        self.num_iter += 1
        recons = output
        input = label

        # 此处使用L1 loss 即MAE loss
        recons_loss = nn.L1Loss()(recons, input)
        # 计算kld loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # 合并，具体两种类型可参照论文
        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss  + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return loss,recons_loss, kld_loss


    # 串联执行，返回结果
    def forward(self, x):

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]


    # 直接使用std和mean重建样本
    def reconstruct(self, x):

        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        z = std + mu
        return  [self.decode(z), mu, log_var]
