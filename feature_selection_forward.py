# _*_ coding: utf-8 _*_
# @Time: 2021/10/2219:51 
# @Author: yuyongsheng
# @Software: PyCharm
# @Description: 南方他克莫司v2.0：(2)使用逐步向前算法筛选重要性特征。

import numpy as np
import pandas as pd

import os
project_path=os.getcwd()

import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import catboost,xgboost
    
# 逐步向前算法
def feature_select_forward(df_model):
    print('-------------------------划分数据集---------------------------')
    # 划分训练集和测试集，比例为8:2
    x = df_model.drop(['TDM检测结果'],axis=1)
    y = df_model['TDM检测结果']
    tran_x, test_x, tran_y, test_y = train_test_split(x, y, test_size=0.2, random_state=5)
    # 使用逐步向前算法训练模型，模型准确度R2
    # 获取测试特征列表
    feature_list = list(tran_x.columns)
    feature_list_copy = feature_list.copy()

    # 保存每次迭代中选出的最佳特征及其当前最佳特征组合r2
    best_feature=[]
    best_r2=[]
    # 利用while循环控制特征个数
    i=1
    while i <= len(feature_list_copy):
        record_feature=[] # 记录测试的特征组合
        record_r2=[]  # 记录测试的特征组合的r2
        # 用for循环控制模型训练轮数
        for j in feature_list:
            feature_test=best_feature.copy()
            feature_test.append(j)
            df_feature=tran_x[feature_test]
            # 训练catboost模型
            model = catboost.CatBoostRegressor(iterations=300, learning_rate=0.2, loss_function='RMSE', random_state=3)
            # model = xgboost.XGBRegressor(iterations=300, learning_rate=0.2, loss_function='RMSE', random_state=3)
            model.fit(df_feature, tran_y)
            # 评价模型性能
            predictions=model.predict(test_x[feature_test])  # xgboost需要变量名称相同
            r2=r2_score(test_y,predictions)
            # 记录特征搭配及其相应的r2
            record_feature.append(feature_test)
            record_r2.append(r2)
        # 评估特征搭配
        r2_max = max(record_r2)
        r2_max_index = record_r2.index(r2_max)
        feature_test_max = record_feature[r2_max_index]
        # 保存本轮迭代选出的最佳单个特征及最优特征组合r2
        best_feature = feature_test_max
        best_r2.append(r2_max)
        # 从测试特征中删除当前最优单个特征
        feature_list.remove(best_feature[-1])

        i +=1
    # 保存模型测试和测试结果到本地文件
    df_feature_test=pd.DataFrame({'特征':best_feature,'r2':best_r2})
    writer = pd.ExcelWriter(project_path + '/result/df_逐步向前特征测试结果.xlsx')
    df_feature_test.to_excel(writer)
    writer.save()

    # 根据逐步向前测试结果筛选最优特征组合
    best_r2_value=max(best_r2)
    best_r2_value_index=best_r2.index(best_r2_value)
    df_feature_selection=df_feature_test.iloc[:best_r2_value_index+1,:]
    writer = pd.ExcelWriter(project_path + '/result/df_逐步向前特征选择结果.xlsx')
    df_feature_selection.to_excel(writer)
    writer.save()

if __name__=='__main__':
    # 读取建模数据集
    print('--------------------读取数据------------------------------')
    df_model = pd.read_excel(project_path + '/data/data_model_from_jinyuan.xlsx')
    df_model = df_model.drop(['patient_id','new_id'],axis=1)
    # 重新排序建模数据集df_model顺序
    df_model=df_model.sort_values(by=['TDM检测结果'],ascending=True)
    df_model=df_model.reset_index(drop=True)
    print(df_model.shape)
    print(df_model.columns)
    feature_select_forward(df_model)