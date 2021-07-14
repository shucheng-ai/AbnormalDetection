import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,TimeSeriesSplit
import datetime
import h5py
# 画出roc_auc曲线
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from matplotlib import pylab as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score
import seaborn as sns
from models import *
from data import *
from utils import *

from config import *


# 加载训练数据，切分为训练和验证部分
with h5py.File(train_data_path, 'r') as hf:
    X_train = hf['X_train'].value 
    X_test = hf['X_val'].value 
    y_train = hf['y_train'].value 
    y_test = hf['y_val'].value 


# 加载测试数据
with h5py.File(test_data_path, 'r') as hf:
    X_t= hf['X'].value 
    y_t = hf['y'].value 


print('y_t_mean',y_t.mean())
print('y_t_sum',y_t.sum())

print(X_train.shape)


# 得到所有特征列名
columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label']

df = pd.read_csv(ten_percent_path,header=None,names=columns)

# 只是用http类型数据进行训练和验证
df = df[df['service']=='http']



# In[47]:

# 去除所有类目类型变量特征
features = df.columns.tolist()
features.remove('label')
new_features = []
for f in features:
    if df[f].dtype!='object':
        new_features.append(f)
features = new_features
print('n_features',len(new_features))


y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

## 创建训练和验证数据迭代器
batches = [[x,y] for x,y in zip(X_train,y_train)]
print('n_samples train',len(batches))
train_iter = build_iterator(batches,batch_size=batch_size)
batches = [[x,y] for x,y in zip(X_test,y_test)]
print('n_samples val',len(batches))
dev_iter = build_iterator(batches,batch_size=batch_size,shuffle=False)


# 初始化模型

# model = AE_LSTM(input_dim=X_train.shape[2],hidden_dim=128,max_len=window_size,loss_type=loss_type,beta=beta,gamma=gamma).to(device)

# model = AE_ATT_TCN(input_dim=X_train.shape[2],hidden_dim=hidden_dim,max_len=window_size,loss_type=loss_type,beta=beta,gamma=gamma).to(device)

# model = AE_ATT_LSTM(input_dim=X_train.shape[2],hidden_dim=128,max_len=window_size,loss_type=loss_type,beta=beta,gamma=gamma,recon_loss_function=recon_loss_function).to(device)

recon_loss_function = nn.MSELoss()
# model = AE_ATT_LightConv(input_dim=X_train.shape[2],hidden_dim=hidden_dim,max_len=window_size,loss_type=loss_type,beta=beta,gamma=gamma,recon_loss_function=recon_loss_function).to(device)
model = AE_ATT_LightTCN(input_dim=X_train.shape[2],hidden_dim=hidden_dim,max_len=window_size,loss_type=loss_type,beta=beta,gamma=gamma,recon_loss_function=recon_loss_function).to(device)

# model = AE_ATT_BiLightTCN(input_dim=X_train.shape[2],hidden_dim=hidden_dim,max_len=window_size,loss_type=loss_type,beta=beta,gamma=gamma).to(device)




# early stoping所需要的的提升批次数
require_improvement = require_improvement_epoch*int(X_train.shape[0]/128)

# 加载模型
if load_model:
    model.load_state_dict(torch.load(save_path))


# 当前时间
start_time = time.time()
# 模型开始训练
model.train()

# 初始化Adam作为梯度下降算法，赋值模型的所有参数
optimizer = optim.Adam([
    {'params':[ param for name, param in model.named_parameters()],'lr':learning_rate},
], lr=learning_rate)


total_batch = 0  # 记录进行到多少batch
dev_best_loss = float('inf')
last_improve = 0  # 记录上次验证集loss下降的batch数
flag = False  # 记录是否很久没有效果提升

model.train()

# 初始化FGM
fgm = FGM(model)


# 选择是否训练模型，开始训练
if train_model:
    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, (trains,labels) in tqdm(enumerate(train_iter)):
            labels = labels.view(-1)

            outputs, mu, log_var = model(trains)
            model.zero_grad()
            
            
            loss,recon_loss,kld_loss = model.loss_function(outputs,trains,mu,log_var,kld_weight=kld_weight)

            loss.backward()

            # 是否对抗训练模型
            if use_fgm:        
                fgm.attack(emb_name=emb_name,epsilon=epsilon)
                outputs_adv, mu_adv, log_var_adv = model(trains)
                loss_adv,recon_loss,kld_loss = model.loss_function(outputs_adv,trains,mu_adv,log_var_adv,kld_weight=kld_weight)

                loss_adv.backward()
                fgm.restore(emb_name=emb_name)
                
            optimizer.step()
            
            
            # 是否训练完一轮
            if total_batch % int(X_train.shape[0]/batch_size) == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.detach().cpu()
                predic = outputs.detach().cpu()
                train_acc = 0
                dev_loss,dev_recon_loss,dev_kld_loss = evaluate(model, dev_iter,kld_weight=kld_weight)
                dev_acc = 0

                # 当前loss小于最优loss时，保存模型且赋值为新的最优loss
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                            
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.4},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.4},  Val RECON: {4:>5.4},  Val KLD: {5:>5.4},  Time: {6} {7}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_recon_loss,dev_kld_loss, time_dif, improve))
                model.train()
            total_batch += 1

            if total_batch - last_improve > require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


# 加载测试集

batches = [[x,y] for x,y in zip(X_t,y_t)]
test_iter = build_iterator(batches,batch_size=batch_size,shuffle=False)
test_loss,test_recon_loss,test_kld_loss = evaluate(model, test_iter,kld_weight=kld_weight)
print('test_loss,test_recon_loss,test_kld_loss',test_loss,test_recon_loss,test_kld_loss)

print('len yt',len(y_t))
batches = [[x,y] for x,y in zip(X_t,y_t)]
print('n_samples test',len(batches))
test_iter = build_iterator(batches,batch_size=batch_size,shuffle=False)

# 输出预测结果
y,y_pred = predict(model, test_iter,attack=attack)

print('len y_pred_t',len(y_pred))


# 把预测error归一化到0~1，方便搜索最优threshold，进而计算recall和precision以及f1
minmax = MinMaxScaler()
y_pred = minmax.fit_transform(y_pred.reshape(-1,1)).reshape(-1)

thresholds = range(0,1000)


best_f1 = -np.inf
l = []
for t in thresholds:
    t = t/1000
    v = (t,f1_score(y,(y_pred>t).astype(np.int)))
    l.append(v)




l = sorted(l,key=lambda x:x[1])

# 误差分布图
sns.distplot(y_pred,bins=100)
plt.show()

best_threshold = l[-1][0]
best_f1 = l[-1][1]

# best_threshold = 0.5
y_c = (y_pred>best_threshold).astype(np.int)
p = precision_score(y,y_c)
r = recall_score(y,y_c)
f1 = f1_score(y,y_c)
print('precision_score',p)
print('recall_score',r)
print('f1_score',f1)
print('best_threshold: %s, best_f1: %s'%(best_threshold,best_f1))



score = roc_auc_score(y,y_pred)
print('auc: %s'%score)

# 画出roc曲线
fpr, tpr, _ = roc_curve(y, y_pred, pos_label=1)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

# 画出混淆矩阵
cm = confusion_matrix(y, y_c)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()


sns.scatterplot(
    np.arange(len(y_pred)),
    y_pred,
    hue = ['normal' if i==0 else 'abnormal' for i in y_c],
    palette=['blue','red'],legend='full')
plt.axhline(y=best_threshold,linestyle='--',label='threshold')
plt.xlabel('samples')
plt.ylabel('reconstraction error')
plt.legend()
plt.grid()
plt.show()
