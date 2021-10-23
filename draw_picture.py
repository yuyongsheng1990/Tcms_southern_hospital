# _*_ coding: utf-8 _*_
# @Time: 2021/10/2320:23 
# @Author: yuyongsheng
# @Software: PyCharm
# @Description:南方他克莫司v2.0：(3)绘制逐步向前算法挑选特征r2折线图

import pandas as pd
import numpy as np
import os
project_path = os.getcwd()

# 判断文件路径是否存在，如果不存在则创建该路径
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

df_feature=pd.read_excel(project_path + '/result/df_逐步向前特征选择结果.xlsx')
df_feature=df_feature.drop(['Unnamed: 0'],axis=1)
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
plt.figure(figsize=(10,8))
plt.plot(list(range(df_feature.shape[0])), df_feature['r2'],
         color='indianred', label='R2')
# 展示横坐标
x_list=list(df_feature['特征'])
plt.xticks(list(range(len(x_list))),x_list,rotation=285)
plt.legend(bbox_to_anchor=(1.1,1))  # 显示图例
# 判断图片保存路径是否存在，否则创建
jpg_path = project_path + "/jpg"
mkdir(jpg_path)
plt.savefig(jpg_path + "/逐步向前特征选择r2折线图.png", dpi=300)
plt.clf()  # 删除前面所画的图

# MSE、RMSE折线图
plt.plot(list(range(df_feature.shape[0])), df_feature['MSE'],
         color='b', label='MSE')
plt.plot(list(range(df_feature.shape[0])), df_feature['RMSE'],
         color='indianred', label='RMSE')
# 展示横坐标
x_list=list(df_feature['特征'])
plt.xticks(list(range(len(x_list))),x_list,rotation=285)
plt.legend(bbox_to_anchor=(1.1,1))  # 显示图例
# 判断图片保存路径是否存在，否则创建
jpg_path = project_path + "/jpg"
mkdir(jpg_path)
plt.savefig(jpg_path + "/逐步向前特征选择MSE和RMSE折线图.png", dpi=300)
plt.clf()  # 删除前面所画的图
