# file config
use_all = True # 是否使用全量数据
# full_path = 'kddcup.data\\kddcup.data.corrected' # 全量数据路径
full_path = 'SKAB-master\\data\\anomaly-free\\anomaly-free.csv'
ten_percent_path  = 'kddcup.data_10_percent\\kddcup.data_10_percent_corrected' # 10%数据路径
# test_path = 'corrected\\corrected' # 验证集路径
test_path = 'SKAB-master\\data\\valve1\\0.csv' # 验证集路径

window_size = 30

train_data_path = 'kdd99.h5' # 训练数据文件目录
test_data_path = 'test_kdd99.h5' # 验证数据文件目录

scaler_path = 'scaler.pkl' # scaler文件目录

loss_type = 'B' # vbae类型，推荐H
save_path = 'vbae_AE_ATT_LightConv.h5'

## train config
train_model = True # 是否训练模型
load_model = False # 是否加载训练好模型
use_fgm = False # 控制是否使用FGM对抗攻击的方式增加模型鲁棒性
attack = False
epsilon = 0.0001




## model config
emb_name = ['log_var','mu']

learning_rate = 0.001
hidden_dim = 64
beta = 10. # beta vae参数
gamma = 1000. # beta vae参数

batch_size = 128
require_improvement_epoch = 20
num_epochs = 2000
kld_weight = 0.0016 # kld loss权重

