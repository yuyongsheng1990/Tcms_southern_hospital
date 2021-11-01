



import pandas as pd
import numpy as np
import os
project_path=os.getcwd()
# 津源代码
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import math
import matplotlib.pyplot as plt
# 判断文件路径是否存在，如果不存在则创建该路径
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

df = pd.read_excel(project_path+'/data/v2.0/建模用数据集（未插补）20210525-3.xlsx')
if 'Unnamed: 0' in df.columns:
    df = df.drop(['Unnamed: 0'], axis=1)
continuous_list = [
  '年龄', '身高(cm)', '体重(kg)', 'BMI', '他克莫司频次', '他克莫司单次剂量', '他克莫司日剂量',
  'C反应蛋白_检测结果', '丙氨酸氨基转移酶_检测结果', '中性粒细胞总数_检测结果', '低密度脂蛋白胆固醇_检测结果',
  '凝血酶原时间比率_检测结果', '天门冬氨酸氨基转移酶_检测结果', '尿素_检测结果', '尿酸_检测结果',
  '平均RBC血红蛋白浓度_检测结果', '平均红细胞体积_检测结果', '平均红细胞血红蛋白量_检测结果', '平均血小板容积_检测结果',
  '总胆固醇_检测结果', '总胆红素_检测结果', '总蛋白_检测结果', '极低密度脂蛋白胆固醇_检测结果', '活化部分凝血活酶时间_检测结果',
  '淋巴细胞总数_检测结果', '球蛋白_检测结果', '甘油三酯_检测结果', '白/球比值_检测结果', '白细胞计数_检测结果',
  '白蛋白_检测结果', '直接胆红素_检测结果', '红细胞比积测定_检测结果', '肌酐_检测结果', '葡萄糖_检测结果',
  '血小板计数_检测结果', '血浆D-二聚体测定_检测结果', '血红蛋白测定_检测结果', '转氨酶比值_检测结果', '间接胆红素_检测结果',
  '非高密度脂蛋白胆固醇_检测结果', '高密度脂蛋白胆固醇_检测结果', '乳酸脱氢酶_检测结果', '心型肌酸激酶_检测结果',
  '肌酸激酶_检测结果', '尿白细胞(仪器定量)_检测结果', '尿红细胞(仪器定量)_检测结果', 'TDM检测结果'
]
#连续变量取log
df_final_10_1 = df.copy()
#df_final_11_1 = df_final_11.copy()
for col in continuous_list:
    df_final_10_1[col] = df_final_10_1[col].apply(lambda x: np.log(x) if x > 0 else np.nan if x!=x else 0)
def model_xy(model):
    x = model[model.columns[2:-1]]
    y = model['TDM检测结果']
    return x, y
col=['身高(cm)', '他克莫司日剂量', '其他免疫抑制剂', '低密度脂蛋白胆固醇_检测结果', '平均红细胞体积_检测结果', '平均红细胞血红蛋白量_检测结果',
     '白细胞计数_检测结果', '直接胆红素_检测结果', '红细胞比积测定_检测结果']
df_model_4 = df_final_10_1.copy()
x4, y4 = model_xy(df_model_4)
all_all_results = []
for j in range(1,52):
    for xy in [[x4, y4]]:
        train_x, test_x, train_y, test_y = train_test_split(xy[0],xy[1],test_size=0.2,random_state=78)
    # 津源xgboost模型
    sfs = SFS(xgb.XGBRegressor(max_depth=5,
                              learning_rate=0.01,
                              n_estimators=500,
                              min_child_weight=0.5,
                              eta=0.1,
                              gamma=0.5,
                              reg_lambda=10,
                              subsample=0.5,
                              colsample_bytree=0.8,
                              nthread=4,
                              scale_pos_weight=1),
             k_features=j,
             forward=True,
             floating=False,
             verbose=2,
             scoring='r2',
             cv=3)

    sfs = sfs.fit(train_x, train_y)
    # 逐步向前筛选结果，包括特征个数，最优特征组合及其r2
    sfs_result = sfs.subsets_
    print(sfs_result)
    df_sfs = pd.DataFrame(sfs_result)
    # DataFrame转置
    df_sfs_T=pd.DataFrame(df_sfs.values.T,index=df_sfs.columns,columns=df_sfs.index)
    df_sfs_T=df_sfs_T.reset_index(drop=True)
    # 保存逐步向前筛选结果
    r2_list=list(df_sfs_T['avg_score'])
    feature_list=list(df_sfs_T['feature_names'])

    # 根据逐步向前测试结果筛选最优特征组合
    r2_max=max(r2_list)
    print(r2_max)
    r2_max_index=r2_list.index(r2_max)
    df_feature_select=df_sfs_T.iloc[r2_max_index:r2_max_index+1,:]
    all_all_results.append(df_feature_select)
df_feature_select=all_all_results[0]
for j in range(1,len(all_all_results)):
    df_feature_select=pd.concat([df_feature_select,all_all_results[j]],axis=0)
df_feature_select=df_feature_select.reset_index(drop=True)
# 保存模型测试和测试结果到本地文件
writer = pd.ExcelWriter(project_path + '/data/v2.0/df_逐步向前特征测试结果.xlsx')
df_feature_select.to_excel(writer)
writer.save()