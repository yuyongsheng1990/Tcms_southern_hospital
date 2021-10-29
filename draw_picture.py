# _*_ coding: utf-8 _*_
# @Time: 2021/10/2320:23 
# @Author: yuyongsheng
# @Software: PyCharm
# @Description:南方他克莫司v2.0：(4)绘制逐步向前算法挑选特征r2折线图

import pandas as pd
import numpy as np
import os
project_path = os.getcwd()

# 判断文件路径是否存在，如果不存在则创建该路径
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

df_model=pd.read_excel(project_path + '/data/v2.0/data_model_from_jinyuan.xlsx')
if 'Unnamed: 0' in df_model.columns:
    df_model=df_model.drop(['Unnamed: 0'],axis=1)

# for i in df_model.columns:
#     if df_model[i].nunique() > 5:
#         df_model[i]=df_model[i].apply(lambda x: np.log(x) if x>1 else x)
df_feature=pd.read_excel(project_path + '/data/v2.0/df_逐步向前特征测试结果.xlsx')
if 'Unnamed: 0' in df_feature.columns:
    df_feature=df_feature.drop(['Unnamed: 0'],axis=1)
df_model=df_model.sort_values(['TDM检测结果'])
df_model=df_model.reset_index(drop=True)
df_feature=df_feature.reset_index(drop=True)
# print(df_model.shape)
# print(df_model.columns)

print('-------------------------划分数据集---------------------------')
from auto_ml import Predictor
from sklearn.model_selection import train_test_split
# 划分训练集和测试集，比例为8:2
x = df_model.drop(['TDM检测结果'],axis=1)
y = df_model['TDM检测结果']
tran_x, test_x, tran_y, test_y = train_test_split(x, y, test_size=0.2, random_state=5)
# 计算R2和MSE、RMSE
import xgboost
import lightgbm
import catboost
from sklearn.metrics import r2_score
import re
# XGBoost模型
r2_list=[]
mse_list=[]
rmse_list=[]
for i in range(0,30):
    feature_list=df_feature.loc[i,'特征'].split('\'')
    feature_list=feature_list[1:-1:2]

    tran_x_select=tran_x[feature_list]
    test_x_select=test_x[feature_list]
    xgb_model=xgboost.XGBRegressor(max_depth=5,
                            learning_rate=0.01,
                            n_estimators=500,
                            min_child_weight=0.5,
                            eta=0.1,
                            gamma=0.5,
                            reg_lambda=10,
                            subsample=0.5,
                            colsample_bytree=0.8,
                            nthread=4,
                            scale_pos_weight=1)

    xgb_model.fit(tran_x_select,tran_y)
    predictions=xgb_model.predict(tran_x_select)

    # 计算r2、MSE和RMSE
    from sklearn.metrics import mean_squared_error  # 均方误差
    from sklearn.metrics import mean_absolute_error  # 平方绝对误差
    from sklearn.metrics import r2_score  # R square
    r2 = r2_score(tran_y,predictions)
    r2_list.append(r2)
    print('r2: ',r2)
    mse=mean_squared_error(tran_y,predictions)
    mse_list.append(mse)
    print('MSE: ',mse)
    rmse=mse ** 0.5
    rmse_list.append(rmse)
    print('RMSE',rmse)
    mae=mean_absolute_error(tran_y,predictions)
    print('MAE: ',mae)
# 画图
print('-----------------------画图---------------------------')
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  ##绘图显示中文
mpl.rcParams['axes.unicode_minus'] = False

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rc

rc('mathtext', default='regular')

# r2折线图
# 分辨率参数-dpi，画布大小参数-figsize
# plt.figure(figsize=(10,8))
plt.plot(list(range(len(r2_list))), r2_list,
         color='indianred', label='R2')
plt.xlabel('Number of Ranked features')
# 展示横坐标
# x_list=list(df_feature['特征'])
# plt.xticks(list(range(len(x_list))),x_list,rotation=285)
# plt.legend(bbox_to_anchor=(1.1,1))  # 显示图例
# 判断图片保存路径是否存在，否则创建
jpg_path = project_path + "/jpg"
mkdir(jpg_path)
plt.savefig(jpg_path + "/逐步向前特征选择r2折线图.png", dpi=300)
plt.clf()  # 删除前面所画的图

# MSE、RMSE折线图
plt.plot(list(range(len(mse_list))), mse_list,
         color='b', label='MSE')
plt.plot(list(range(len(rmse_list))), rmse_list,
         color='indianred', label='RMSE')
plt.xlabel('Number of Ranked features')

# 展示横坐标
# x_list=list(df_feature['特征'])
# plt.xticks(list(range(len(x_list))),x_list,rotation=285)
# plt.legend(bbox_to_anchor=(1.1,1))  # 显示图例
# 判断图片保存路径是否存在，否则创建
jpg_path = project_path + "/jpg"
mkdir(jpg_path)
plt.savefig(jpg_path + "/逐步向前特征选择MSE和RMSE折线图.png", dpi=300)
plt.clf()  # 删除前面所画的图
