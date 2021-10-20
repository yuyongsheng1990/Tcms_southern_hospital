# _*_ coding: utf-8 _*_
# @Time: 2021/10/1821:24 
# @Author: yuyongsheng
# @Software: PyCharm
# @Description: 南方医学他克莫司:(2) 数据建模，并画图

import pandas as pd
import numpy as np

import re
import os
project_path = os.getcwd()

import sys

from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
from auto_ml.utils_models import load_ml_model

from sklearn.metrics import r2_score

# 判断文件路径是否存在，如果不存在则创建该路径
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

def main():
    print('--------------------读取数据------------------------------')
    df_model = pd.read_excel(project_path + '/result/df_15_插补tdm检测7天内最近的其他联合用药.xlsx')
    df_model = df_model.drop(['Unnamed: 0'], axis=1)
    # 重新排序建模数据集df_model顺序
    df_model=df_model.sort_values(by=['test_result'],ascending=True)
    df_model=df_model.reset_index(drop=True)
    print(df_model.shape)
    print(df_model['patient_id'].nunique())

    print(df_model.columns)
    # 删除建模数据中不需要的字段，patient_id,long_d_order,drug_name,drug_spec,dosage,frequency,start_datetime,end_datetime,test_date,project_name,is_normal
    df_model=df_model.drop(['patient_id','long_d_order','drug_name','drug_spec','start_datetime','end_datetime','test_date','project_name','is_normal'],axis=1)

    # 分段、归一化处理建模数据
    from sklearn.model_selection import train_test_split
    print('-------------------------划分数据集---------------------------')
    # 划分训练集和测试集，比例为8:2
    x = df_model
    y = df_model['test_result']
    tran_x, test_x, tran_y, test_y = train_test_split(x, y, test_size=0.25, random_state=5)

    print(tran_x.columns)

    column_descriptions = {'test_result': 'output'}
    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
    ml_predictor.train(tran_x, model_names=['XGBRegressor'])

    file_name = ml_predictor.save()
    trained_model = load_ml_model(file_name)

    # A list of dictionaries
    # A single dictionary (optimized for speed in production evironments)
    predictions = ml_predictor.predict(test_x)
    print(predictions)
    r2 = r2_score(predictions,test_y)
    print(r2)

    df_predictions= pd.DataFrame(data={'真实值':test_y,'预测值':predictions})
    writer = pd.ExcelWriter(project_path + '/result/df_16_他克莫司血药浓度测试结果.xlsx')
    df_predictions.to_excel(writer)
    writer.save()

    from pylab import mpl

    mpl.rcParams['font.sans-serif'] = ['SimHei']  ##绘图显示中文
    mpl.rcParams['axes.unicode_minus'] = False

    from matplotlib import pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib import rc

    rc('mathtext', default='regular')

    sub_axix = filter(lambda x: x % 20 == 0, list(range(test_x.shape[0])))
    plt.plot(list(range(test_x.shape[0])), np.array(list(predictions)),
             color=(0.32941176470588235, 0.7294117647058823, 0.7490196078431373),
             label='The Predictive Value of test result')
    # plt.plot(sub_axix, test_acys, color='red', label='testing accuracy')
    plt.plot(list(range(test_x.shape[0])), np.array(list(test_x['test_result'])), color='indianred',
             label='The True Value of test result')
    # plt.plot(list(range(test_x.shape[0])), thresholds, color='blue', label='threshold')
    plt.legend(bbox_to_anchor=(1.1, 1))  # 显示图例

    plt.xlabel('Number of Events(unit)')
    plt.ylabel('Tac Daily Dose(mg)')
    # plt.show()
    # 判断图片保存路径是否存在，否则创建
    jpg_path = project_path + "/jpg"
    mkdir(jpg_path)
    plt.savefig(jpg_path + "/他克莫司血药浓度测试集折线图_15_15_is.jpg", dpi=300)
    plt.clf()  # 删除前面所画的图

if __name__=='__main__':
    main()