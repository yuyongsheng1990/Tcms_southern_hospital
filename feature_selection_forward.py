# _*_ coding: utf-8 _*_
# @Time: 2021/10/2219:51 
# @Author: yuyongsheng
# @Software: PyCharm
# @Description: 南方他克莫司v2.0：(2)使用逐步向前算法筛选重要性特征。
import itertools

import numpy as np
import pandas as pd

import os
project_path=os.getcwd()

import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error,mean_absolute_error  # 均方误差
import warnings
warnings.filterwarnings("ignore")
import catboost,xgboost

# 判断文件路径是否存在，如果不存在则创建该路径
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

# 逐步向前算法
def feature_select_forward(df_model):

    from sklearn.preprocessing import StandardScaler
    discrete_list=['gender','糖皮质激素','质子泵','钙离子阻抗剂','其他免疫抑制剂','克拉霉素或阿奇霉素']
    continuous_list = [x for x in df_model.columns if x not in discrete_list]
    # continuous_list.remove('TDM检测结果')

    print('-------------------------划分数据集---------------------------')
    # 划分训练集和测试集，比例为8:2
    x = df_model.drop(['TDM检测结果'],axis=1)
    y = df_model['TDM检测结果']
    # 连续变量归一化
    for col in continuous_list:
        df_model[col] = df_model[col].apply(lambda x: np.log(x) if x > 0 else np.nan if x != x else 0)
    # for i in continuous_list:
    #     if df_model[i].nunique() > 5:
    #         df_model[i]=df_model[i].apply(lambda x: np.log(x) if x>10 else x)
    tran_x, test_x, tran_y, test_y = train_test_split(x, y, test_size=0.2, random_state=5)
    # 归一化
    # std = StandardScaler()
    # tran_x_std = std.fit_transform(tran_x)
    # 调用mlxtend包中的逐步向前算法，模型准确度指标-R2.
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
    import xgboost as xgb
    import matplotlib.pyplot as plt

    # k_features,Number of features to select,where k_features < the full feature set.
    # 基于当前所有变量中做逐步向前，从k=1一直迭代到k个特征
    for i in range(1,30):
        sfs = SFS(xgb.XGBRegressor(max_depth=5,
                                learning_rate=0.1,
                                n_estimators=500,
                                min_child_weight=0.5,
                                eta=0.1,
                                gamma=0.5,
                                reg_lambda=10,
                                subsample=0.5,
                                colsample_bytree=0.8,
                                nthread=4,
                                scale_pos_weight=1),
        # sfs = SFS(catboost.CatBoostRegressor(n_estimators=500,learning_rate=0.01),
                  k_features=i,
                  forward=True,
                  floating=False,
                  verbose=2,
                  scoring='r2',
                  cv=3)  # cv表示交叉验证
        sfs = sfs.fit(tran_x, tran_y)
        # 逐步向前筛选结果，包括特征个数，最优特征组合及其r2
        sfs_result = sfs.subsets_
        print(sfs_result)

    df_sfs = pd.DataFrame(sfs_result)
    # DataFrame转置
    df_sfs_T=pd.DataFrame(df_sfs.values.T,index=df_sfs.columns,columns=df_sfs.index)
    df_sfs_T=df_sfs_T.reset_index(drop=True)
    # 保存逐步向前筛选结果
    # # 如果用归一化，将特征名称由索引替换为汉字
    # columns_list=list(df_model.columns)
    # temp_list=list(df_sfs_T['feature_names'])
    # feature_list=[]
    # for i in temp_list:
    #     list_i=[]
    #     for j in i:
    #         names=columns_list[int(j)]
    #         list_i.append(names)
    #     feature_list.append(list_i)
    r2_list=list(df_sfs_T['avg_score'])
    feature_list=list(df_sfs_T['feature_names'])

    df_feature_test=pd.DataFrame({'特征':feature_list,'r2':r2_list})
    df_feature_test=df_feature_test.reset_index(drop=True)
    # 保存模型测试和测试结果到本地文件
    writer = pd.ExcelWriter(project_path + '/data/v2.0/df_逐步向前特征测试结果.xlsx')
    df_feature_test.to_excel(writer)
    writer.save()

    # 根据逐步向前测试结果筛选最优特征组合
    r2_max=max(r2_list)
    print(r2_max)
    r2_max_index=r2_list.index(r2_max)
    df_feature_select=df_feature_test.iloc[r2_max_index]
    writer = pd.ExcelWriter(project_path + '/data/v2.0/df_逐步向前特征选择结果.xlsx')
    df_feature_select.to_excel(writer)
    writer.save()

    feature_extract_list = list(feature_list[r2_max_index])
    feature_extract_list.append('TDM检测结果')
    df_extract = df_model[feature_extract_list]
    writer = pd.ExcelWriter(project_path + '/data/v2.0/df_逐步向前筛选后的建模数据.xlsx')
    df_extract.to_excel(writer)
    writer.save()

    # 逐步向前R2折线图
    metrics=sfs.get_metric_dict()
    # 截取部分字典数据，取30个特征生成逐步向前R2折线图
    metrics_part={key:value for key,value in metrics.items() if key <30}
    fig = plot_sfs(metrics_part, kind='std_err',color='r')
    plt.title('Sequential Forward Selection (R2)')
    plt.grid()
    # plt.show()
    # 判断图片保存路径是否存在，否则创建
    jpg_path = project_path + "/jpg"
    mkdir(jpg_path)
    plt.savefig(jpg_path + "/逐步向前特征选择R2折线图_津源代码.jpg", dpi=300)
    plt.clf()  # 删除前面所画的图

    '''
    # 穷举法。保存每次迭代中选出的最佳特征及其当前最佳特征组合r2
    # 时间太长了！！！！逐步向前 1 min;穷举1天
    # mlx也有穷举法的package,from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
    feature_list = list(tran_x.columns)
    best_num=[]  # 记录特征个数
    best_feature=[]
    best_r2=[]
    best_mae=[]
    best_mse=[]
    best_rmse=[]
    # 穷举法。利用for循环控制特征选择轮数：第一轮测试任意一个特征；第二轮测试任意两个特征组合.....最多到20-30个，因为用药情境中，实际情况不可能有50多种药或检查。
    for i in range(1,20):
        record_feature=[] # 记录测试的特征组合
        record_r2=[]  # 记录测试的特征组合的r2
        record_mae=[]
        record_mse=[]
        record_rmse=[]
        # 使用itertools.combinations()函数生成特征组合。
        feature_combination=itertools.combinations(feature_list,i)
        for j in feature_combination:
            print('第%轮循环' % i)
            feature_test=list(j)
            df_feature=tran_x[feature_test]
            # 使用catboost模型进行逐步向前训练
            # model = catboost.CatBoostRegressor(iterations=300, learning_rate=0.2, loss_function='RMSE', random_state=3)
            # 使用xgbboost模型进行逐步向前训练
            model = xgboost.XGBRegressor(iterations=300, learning_rate=0.2, loss_function='RMSE', random_state=3)
            model.fit(df_feature, tran_y)
            # 评价模型性能
            predictions=model.predict(test_x[feature_test])  # xgboost需要变量名称相同
            r2=r2_score(test_y,predictions)
            mae = mean_absolute_error(test_y, predictions)
            mse = mean_squared_error(test_y, predictions)
            rmse = mse ** 0.5
            # 记录特征搭配及其相应的r2
            record_feature.append(feature_test)
            record_r2.append(r2)
            record_mae.append(mae)
            record_mse.append(mse)
            record_rmse.append(rmse)
        # 评估特征搭配
        r2_max = max(record_r2)
        r2_max_index = record_r2.index(r2_max)
        mae_max=record_mae[r2_max_index]
        mse_max=record_mse[r2_max_index]
        rmse_max=record_rmse[r2_max_index]
        feature_combination_max = record_feature[r2_max_index]
        # 保存本轮迭代选出的最佳单个特征及最优特征组合r2、MSE和RMSE
        best_num.append(i)
        best_feature.append(feature_combination_max)
        best_r2.append(r2_max)
        best_mae.append(mae_max)
        best_mse.append(mse_max)
        best_rmse.append(rmse_max)

    # 保存模型测试和测试结果到本地文件
    df_feature_test=pd.DataFrame({'特征个数':best_num,'特征':best_feature,'r2':best_r2,'MAE':best_mae,'MSE':best_mse,'RMSE':best_rmse})
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
    
    extract_list=df_feature_selection['特征'][0]
    extract_list.append('TDM检测结果')
    df_extract=df_model[extract_list]
    writer = pd.ExcelWriter(project_path + '/data/v2.0/df_逐步向前筛选后的建模数据.xlsx')
    df_extract.to_excel(writer)
    writer.save()
    '''


if __name__=='__main__':
    # 读取建模数据集
    print('--------------------读取数据------------------------------')
    df_model = pd.read_excel(project_path + '/data/v2.0/data_model_from_jinyuan.xlsx')
    # df_model = df_model.drop(['patient_id','new_id'],axis=1)
    # 重新排序建模数据集df_model顺序
    # df_model=df_model.sort_values(by=['TDM检测结果'],ascending=True)
    df_model=df_model.reset_index(drop=True)
    print(df_model.shape)
    print(df_model.columns)
    feature_select_forward(df_model)