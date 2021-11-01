# _*_ coding: utf-8 _*_
# @Time: 2021/10/1821:24 
# @Author: yuyongsheng
# @Software: PyCharm
# @Description: 南方医学他克莫司v1.0:(2) 数据建模，并画图

import pandas as pd
import numpy as np

import re
import os
project_path = os.getcwd()

import sys

from auto_ml.utils_models import load_ml_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 判断文件路径是否存在，如果不存在则创建该路径
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

def main():
    print('--------------------读取数据------------------------------')
    df_model=pd.read_excel(project_path + '/data/v2.0/df_逐步向前筛选后的建模数据.xlsx')
    if 'Unnamed: 0' in df_model.columns:
        df_model=df_model.drop(['Unnamed: 0'],axis=1)
    df_model=df_model[df_model.columns[2:]]
    # 重新排序建模数据集df_model顺序
    # df_model=df_model.sort_values(by=['TDM检测结果'],ascending=True)
    df_model=df_model.reset_index(drop=True)
    print(df_model.shape)
    print(df_model.columns)

    print('-------------------------划分数据集---------------------------')
    from auto_ml import Predictor
    # 划分训练集和测试集，比例为8:2
    x = df_model.drop(['TDM检测结果'],axis=1)
    y = df_model['TDM检测结果']
    tran_x, test_x, tran_y, test_y = train_test_split(x, y, test_size=0.2, random_state=5)

    print('-------------------------训练模型---------------------------')

    # 直接使用xgboost和catboost包，而不是auto_ml
    import xgboost
    import lightgbm
    import catboost
    from sklearn.metrics import r2_score
    # XGBoost模型
    # xgb_model=xgboost.XGBRegressor()
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
    # LightGBM模型
    # xgb_model=lightgbm.LGBMRegressor(iterations=300, learning_rate=0.2, loss_function='RMSE',random_state=3)
    # CatBoost模型
    # xgb_model=catboost.CatBoostRegressor(iterations=300, learning_rate=0.2, loss_function='RMSE',random_state=3)
    xgb_model.fit(tran_x,tran_y)
    predictions=xgb_model.predict(test_x)
    # print(predictions)

    # 计算R2和均方误差MSE
    print('-----------------------计算R2和均方误差MSE---------------------------')

    from sklearn.metrics import mean_squared_error  # 均方误差
    from sklearn.metrics import mean_absolute_error  # 平方绝对误差
    from sklearn.metrics import r2_score  # R square
    # 调用

    r2 = r2_score(test_y,predictions)
    print('r2: ',r2)
    mse=mean_squared_error(test_y,predictions)
    print('MSE: ',mse)
    mae=mean_absolute_error(test_y,predictions)
    print('MAE: ',mae)

    df_predictions= pd.DataFrame(data={'真实值':test_y,'预测值':predictions})
    writer = pd.ExcelWriter(project_path + '/result/df_18_他克莫司TDM模型测试结果.xlsx')
    df_predictions.to_excel(writer)
    writer.save()

    # 画图
    print('-----------------------画图---------------------------')
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']  ##绘图显示中文
    mpl.rcParams['axes.unicode_minus'] = False

    from matplotlib import pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib import rc
    rc('mathtext', default='regular')

    # 散点图
    # axis设置坐标轴的范围
    # plt.axis([-20, 20, 0, 200])
    # x为x轴中坐标x的值，y为y轴中坐标y的值，x与y都是长度相同的数组序列，color为点的颜色，marker为散点的形状，
    # 折线图刻度调小，要不然点都堆到一块了
    ax = plt.gca()
    ax.set_xlim(0,12)
    ax.set_ylim(0,10)
    # plt.scatter(range(len(test_y)),test_y,c='r')
    plt.scatter(test_y,predictions,c='b')
    # 红色参照线
    plt.plot(list(range(test_y.shape[0])), list(range(test_y.shape[0])),color='r')
    # plt.plot(list(range(30)), list(range(30)),color='r')
    plt.xlabel('Number of Events(unit)')
    plt.ylabel('Tac TDM CONC.')
    # plt.show()
    # 判断图片保存路径是否存在，否则创建
    jpg_path = project_path + "/jpg"
    mkdir(jpg_path)
    plt.savefig(jpg_path + "/他克莫司血药浓度测试集散点图v2.0.jpg", dpi=300)
    plt.clf()  # 删除前面所画的图

    '''
    # 重要性
    import catboost,xgboost
    model_boost=xgboost.XGBRegressor()
    model_boost.fit(tran_x,tran_y)
    importance = model_boost.feature_importances_
    print(tran_x.columns)
    print(importance)

    df_importance= pd.DataFrame(data={'特征':tran_x.columns,'重要性评分':importance})
    df_importance['重要性评分']=df_importance['重要性评分'].apply(lambda x: round(x,3))
    df_importance=df_importance.sort_values(['重要性评分'],ascending=False)
    df_importance=df_importance.reset_index(drop=True)
    writer = pd.ExcelWriter(project_path + '/result/df_19_模型重要性评分.xlsx')
    df_importance.to_excel(writer)
    writer.save()

    # SHAP图
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']  ##绘图显示中文
    mpl.rcParams['axes.unicode_minus'] = False
    from matplotlib import rc
    rc('mathtext', default='regular')

    import xgboost as xgb
    import shap
    shap.initjs()  # notebook环境下，加载用于可视化的JS代码

    importance_list_10 = list(df_importance['特征'])[:10]
    tran_x=tran_x[importance_list_10]
    shap_model = xgb.XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=300)
    shap_model.fit(tran_x, tran_y)

    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(tran_x)  # 传入特征矩阵X，计算SHAP值
    # print(shap_values)
    # summarize the effects of all the features
    shap.summary_plot(shap_values, tran_x, plot_size=(20,12))
    # 保存各个变量的shape值的和
    df_shap_values=pd.DataFrame(shap_values)
    shap_list=[]
    for i in range(df_shap_values.shape[1]):
        df_shap_values.iloc[:,i] = df_shap_values.iloc[:,i].apply(lambda x: abs(x))
    for j in range(df_shap_values.shape[1]):
        feature_mean = df_shap_values.iloc[:,j].mean()
        feature_mean = round(feature_mean, 2)
        shap_list.append(feature_mean)

    df_shap = pd.DataFrame({'features':list(tran_x.columns),'shap值':shap_list})
    df_shap = df_shap.sort_values(by=['shap值'], ascending=False)
    df_shap = df_shap.reset_index(drop=True)

    writer = pd.ExcelWriter(project_path + '/result/df_20_shap值排序.xlsx')
    df_shap.to_excel(writer)
    writer.save()
    '''
if __name__=='__main__':
    main()