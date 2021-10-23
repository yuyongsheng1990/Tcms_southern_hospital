# _*_ coding: utf-8 _*_
# @Time: 2021/10/199:12 
# @Author: yuyongsheng
# @Software: PyCharm
# @Description: 南方他克莫司v2.0：(1)统计变量平均数、四分位数

import os
project_path=os.getcwd()
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# 读取津源建模数据
df_model = pd.read_excel(project_path + '/data/data_model_from_jinyuan.xlsx')
# df_model = df_model.drop(['Unnamed: 0'], axis=1)

print('----------------他克莫司用药信息统计----------------------')
# t统计他克莫司TDM数据
tac_test=df_model['TDM检测结果'].describe()
print(tac_test)
# 统计他克莫司频次数据
tac_frequency=df_model['他克莫司频次'].describe()
print(tac_frequency)
# 统计他克莫司单次日剂量数据
tac_dosage=df_model['他克莫司单次剂量'].describe()
print(tac_dosage)
# 统计他克莫司日剂量数据
tac_dosage_day=df_model['他克莫司日剂量'].describe()
print(tac_dosage_day)

print('----------------人口学信息统计----------------------')
# 统计年龄数据
age=df_model['年龄'].describe()
print(age)
# 统计性别数据
gender_1=df_model[df_model['gender']==1].shape[0]
gender_0=df_model[df_model['gender']==0].shape[0]
print(gender_1,gender_1/df_model['gender'].shape[0])
print(gender_0,gender_0/df_model['gender'].shape[0])
# 统计身高数据
height=df_model['身高(cm)'].describe()
print(height)
# 统计体重数据
weight=df_model['体重(kg)'].describe()
print(weight)
# 统计BMI数据
BMI=df_model['BMI'].describe()
print(BMI)

print('----------------联合用药统计----------------------')
# 统计糖皮质激素日剂量
tang_1=df_model[df_model['糖皮质激素']==1].shape[0]
tang_0=df_model[df_model['糖皮质激素']==0].shape[0]
print(tang_1,tang_1/df_model['糖皮质激素'].shape[0])
print(tang_0,tang_0/df_model['糖皮质激素'].shape[0])
# 统计质子泵日剂量
zhizi_1=df_model[df_model['质子泵']==1].shape[0]
zhizi_0=df_model[df_model['质子泵']==0].shape[0]
print(zhizi_1,zhizi_1/df_model['质子泵'].shape[0])
print(zhizi_0,zhizi_0/df_model['质子泵'].shape[0])

# 统计钙离子阻抗剂日剂量
gai_1=df_model[df_model['钙离子阻抗剂']==1].shape[0]
gai_0=df_model[df_model['钙离子阻抗剂']==0].shape[0]
print(gai_1,gai_1/df_model['钙离子阻抗剂'].shape[0])
print(gai_0,gai_0/df_model['钙离子阻抗剂'].shape[0])
# 统计其他免疫抑制剂日剂量
mianyi_1=df_model[df_model['其他免疫抑制剂']==1].shape[0]
mianyi_0=df_model[df_model['其他免疫抑制剂']==0].shape[0]
print(mianyi_1,mianyi_1/df_model['其他免疫抑制剂'].shape[0])
print(mianyi_0,mianyi_0/df_model['其他免疫抑制剂'].shape[0])
# 统计克拉霉素或阿奇霉素日剂量
aqms_1=df_model[df_model['克拉霉素或阿奇霉素']==1].shape[0]
aqms_0=df_model[df_model['克拉霉素或阿奇霉素']==0].shape[0]
print(aqms_1,aqms_1/df_model['克拉霉素或阿奇霉素'].shape[0])
print(aqms_0,aqms_0/df_model['克拉霉素或阿奇霉素'].shape[0])

print('----------------其他检测统计----------------------')
# 统计尿酸数据
n_1=df_model['尿酸_检测结果'].describe()
print('尿酸_检测结果',n_1)
# 统计肌酐
n_2=df_model['肌酐_检测结果'].describe()
print('肌酐_检测结果',n_2)
# 统计C反应蛋白
n_3=df_model['C反应蛋白_检测结果'].describe()
print('C反应蛋白_检测结果',n_3)
# 统计总胆红素
n_4=df_model['总胆红素_检测结果'].describe()
print('总胆红素_检测结果',n_4)
# 统计血红蛋白测定_检测结果
n_5=df_model['血红蛋白测定_检测结果'].describe()
print('血红蛋白测定_检测结果',n_5)
# 统计球蛋白
n_8=df_model['球蛋白_检测结果'].describe()
print('球蛋白_检测结果',n_8)
# 统计中性粒细胞总数
n_6=df_model['中性粒细胞总数_检测结果'].describe()
print('中性粒细胞总数_检测结果',n_6)
# 统计球蛋白
n_8=df_model['高密度脂蛋白胆固醇_检测结果'].describe()
print('高密度脂蛋白胆固醇_检测结果',n_8)

