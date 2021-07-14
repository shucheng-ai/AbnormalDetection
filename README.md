# AbnormalDetection
# AbnormalDetection

本项目使用VAE框架构建时间序列异常检测。

编码器与解码器可选模型包括：
 
 1.BI-RNN

 2.TCN
 
 3.LightCNN
 
 4.LightTCN

 5.BI-LightTCN
 
通过对抗训练和VAE框架增强模型的鲁棒性，增加误差检测过程中正常样本和异常样本的variance

# 执行顺序

数据采用sklab和kdd99进行测试检测，对应配置可在config.py中修改

执行顺序： normal_*.py/test_*.py -> main.py
