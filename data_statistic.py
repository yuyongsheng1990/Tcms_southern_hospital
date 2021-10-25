# _*_ coding: utf-8 _*_
# @Time: 2021/10/199:12 
# @Author: yuyongsheng
# @Software: PyCharm
# @Description: 南方他克莫司v2.0：(1)统计变量平均数、四分位数;(2)正态性检验

import os
project_path=os.getcwd()
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# 读取津源建模数据
df_model = pd.read_excel(project_path + '/data/v2.0/data_model_from_jinyuan.xlsx')
# df_model = df_model.drop(['Unnamed: 0'], axis=1)
'''
print('----------------全体变量数据统计描述----------------------')
# 统计全变量体系各变量的平均数、上下四分位数、缺失率
feature_list=[]
mean_list=[]
up_quarter_list=[]
down_quarter_list=[]
miss_list=[]

for i in df_model.columns:
    data = df_model[i]
    stat_result = pd.DataFrame(data.describe())
    # print(stat_result)
    mean_value=stat_result.loc['mean',i]
    up_quarter=stat_result.loc['25%',i]
    down_quarter=stat_result.loc['75%',i]
    num=stat_result.loc['count',i]
    miss_rate=1-num/df_model.shape[0]
    miss_rate="%.2f%%" % (miss_rate * 100)      # 百分数输出

    feature_list.append(i)
    mean_list.append(round(mean_value,2))
    up_quarter_list.append(round(up_quarter,2))
    down_quarter_list.append(round(down_quarter,2))
    miss_list.append(miss_rate)

df_stat=pd.DataFrame({'特征':feature_list,'平均值':mean_list,'上四分位':up_quarter_list,'下四分位':down_quarter_list,'缺失率':miss_list})
df_stat=df_stat.reset_index(drop=True)

writer=pd.ExcelWriter(project_path+'/data/v2.0/df_全体变量数据统计.xlsx')
df_stat.to_excel(writer)
writer.save()
'''
print('----------------连续变量正态性检验----------------------')
from scipy.stats import shapiro,kstest,skewtest,kurtosistest
import math

feature_list=[]
norm_list=[]  # 正态统计量
p_list=[]  # 正态p值
# kur_list=[]  # 峰度
skew_list=[]  # 偏度

for i in df_model.columns:
    u = df_model[i].mean()
    std = df_model[i].std()
    # 判断变量为连续变量还是分类变量
    if df_model[i].nunique()<=5:
        norm,p=np.nan,np.nan
    # 按样本量-正态性检验
    elif df_model[i].shape[0]> 30:
        norm,p=kstest(df_model[i],'norm',(u,std))
        p=round(p,2)
    else:
        norm,p=shapiro(df_model[i])
        p=round(p,2)
    feature_list.append(i)
    norm_list.append(norm)
    p_list.append(p)
    # 在正态分布下，检验峰度和偏度，峰度和偏度是可选择，取log/归一化一切为了调整数据分布，提升建模效果
    # 现实生活中，绝大多数偏态几乎都是右偏，左偏极少存在。取log的时候，针对整体取log，当x>1时，log(x)
    if p > 0.05:  # 符合正态分布
        skew_value=skewtest(df_model[i])
        # 偏度需要调整的阈值，可以自己设置1、5或10，还是那句话，一切为了建模
        if skew_value >=5:  # 认为右偏严重
            print('右偏严重',i)
            df_model[i]=df_model[i].apply(lambda x: math.log(x) if abs(x) >=1 else x)
df_norm_test=pd.DataFrame({'特征':feature_list,'正态统计量':norm_list,'p值':p_list})
df_norm_test=df_norm_test.reset_index(drop=True)

writer=pd.ExcelWriter(project_path+'/data/v2.0/df_正态性检验结果.xlsx')
df_norm_test.to_excel(writer)
writer.save()

print('----------------相关性检验----------------------')

# 统计模型在测试集上的预测结果
df_model = pd.read_excel(project_path + '/result/df_18_他克莫司TDM模型测试结果.xlsx')
df_model['预测准确率']=(df_model['真实值']-df_model['预测值'])/df_model['真实值']
df_model['预测准确率']=df_model['预测准确率'].apply(lambda x:abs(x))
df_model_10=df_model[df_model['预测准确率'] <=0.1]
df_model_20=df_model[df_model['预测准确率'] <=0.2]
df_model_30=df_model[df_model['预测准确率'] <=0.3]

print(df_model_10.shape[0],df_model_10.shape[0]/df_model.shape[0])
print(df_model_20.shape[0],df_model_20.shape[0]/df_model.shape[0])
print(df_model_30.shape[0],df_model_30.shape[0]/df_model.shape[0])