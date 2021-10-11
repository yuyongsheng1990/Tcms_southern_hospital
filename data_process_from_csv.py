# _*_ coding: utf-8 _*_
# @Time: 2021/10/913:56 
# @Author: yuyongsheng
# @Software: PyCharm
# @Description: 南方医学他克莫司:(1) 从csv文件中提取数据，进行数据处理

import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
from scipy import stats

import os
import re
import datetime

project_path = os.getcwd()


def str_to_datatime(x):
    try:
        a = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        return a
    except:
        return np.NaN


# 读取保存的csv数据
print('------------------------读取csv数据--------------------------------------')
diagnose_record = pd.read_csv(project_path + '/data/df_diagnostic.csv')
diagnose_content = pd.read_csv(project_path + '/data/diagnose_content.csv')
doctor_order = pd.read_csv(project_path + '/data/df_doctor_order.csv')
test_record = pd.read_csv(project_path + '/data/df_test_record.csv')
test_result = pd.read_csv(project_path + '/data/df_result.csv')
surgery_record = pd.read_csv(project_path + '/data/df_surgical.csv')
patient_vital = pd.read_csv(project_path + '/data/patient_vital.csv')

df_figure = pd.read_excel(project_path + '/data/df_figure.xlsx')  # 身高、体重
df_gender = pd.read_csv(project_path + '/data/df_patient_info.csv')  # 性别
df_age = pd.read_csv(project_path + '/data/inp_record.csv')  # 年龄

print(len(np.unique(diagnose_record['patient_id'])))  # 8508
print(len(np.unique(diagnose_content['patient_id'])))  # 8508
print(len(np.unique(doctor_order['patient_id'])))  # 1079
print(len(np.unique(df_gender['patient_id'])))  # 8510
print(len(np.unique(test_record['patient_id'])))  # 3193
print(len(np.unique(surgery_record['patient_id'])))  # 295

# 从test_record和test_result文件中提取有效和有用字段
# diagnose_record = diagnose_record[['diagnostic_record_id','patient_id', 'record_date', 'diagnostic_content']]
diagnose_content = diagnose_content.drop(['Unnamed: 0'], axis=1)
doctor_order = doctor_order[[ 'patient_id', 'long_d_order', 'drug_name', 'amount', \
                             'drug_spec', 'dosage', 'frequency', 'start_datetime', 'end_datetime']]
test_record = test_record[['test_record_id', 'patient_id', 'test_date']]  # (361590,3)
test_result = test_result[['test_result_id', 'test_record_id', 'project_name', 'test_result', 'is_normal']]  # (2741436, 5)
# surgery_record = surgery_record[['surgical_record_id', 'patient_id', 'surgery_date', 'surgery_name']]

print(test_record.shape)  # (361590, 3)
print(test_result.shape)  # (2741136, 5)

# 编程逻辑：可以正向，由他克莫司用药筛选tdm检测；也可以反向，由tdm检测筛选他克莫司用药。我在本项目中取反向逻辑
# 需求1：提取有自身免疫疾病（主要包括狼疮性肾炎、系统性红斑狼疮、肾病综合征）的他克莫司tdm检测记录的病人
print('---------------------1.提取有自身免疫疾病的病人id--------------------')
# 提取有自身免疫疾病的病人id
diagnose_content_disease=diagnose_content[(diagnose_content['diagnostic_content'].str.contains('狼疮性肾炎'))|
                                          (diagnose_content['diagnostic_content'].str.contains('系统性红斑狼疮'))|
                                          (diagnose_content['diagnostic_content'].str.contains('肾病综合征'))]
diagnose_content_disease=diagnose_content_disease.reset_index(drop=True)
print(diagnose_content_disease.shape)  # (546,2)
print(len(np.unique(diagnose_content_disease['patient_id'])))  # 546

# 需求2：提取有自身免疫疾病病人的他克莫司血药浓度（TDM）检测记录，tdm检测前7天内有他克莫司医嘱的数据
print('---------------------2.提取自身免疫疾病病人的他克莫司tdm检测和用药数据--------------------')
# 提取自身免疫疾病病人的检测记录
disease_list=[]
for i in np.unique(diagnose_content_disease['patient_id']):
    temp = test_record[test_record['patient_id']==i]
    disease_list.append(temp)
test_record_disease=disease_list[0]
for j in range(1,len(disease_list)):
    test_record_disease=pd.concat([test_record_disease,disease_list[j]],axis=0)
test_record_disease=test_record_disease.reset_index(drop=True)
print(test_record_disease.shape)  # (28988,3)
print(len(np.unique(test_record_disease['patient_id'])))  # 488

# 根据test_record_id合并自身免疫病人的test_record_disease和test_result
test_record_result = pd.merge(test_record_disease, test_result, on=['test_record_id'], how='inner')  # (2741436, 7)
# 检验结果按照病人id排序
test_record_result = test_record_result.sort_values(['patient_id'], ascending=1)
test_record_result = test_record_result.reset_index(drop=True)

print(test_record_result.shape)  # (209256,7)
print(len(np.unique(test_record_result['patient_id'])))  # 488

# 提取包含他克莫司tdm检测记录的test_result。
test_record_result_tdm = test_record_result[test_record_result['project_name'].str.contains('他克莫司药物浓度监测')]  # 无重复数据
test_record_result_tdm = test_record_result_tdm.reset_index(drop=True)

print(test_record_result_tdm.shape)  # (987, 7)，test_date为空的数据
print(len(np.unique(test_record_result_tdm['patient_id'])))  # 349

# 按需求3，只提取他克莫司胶囊的用药数据
# 避免memoryerror, 先从doctor_order中筛选出有他克莫司医嘱的patient_id，再进行合并
drug_tcms = doctor_order[(doctor_order['drug_name'].str.contains('他克莫司'))]
# 但这里面包含他克莫司检测(药材科)、他克莫司用药基因(药材科)等无效数据，取反~
# drug_tcms=drug_tcms[~ drug_tcms['drug_name'].str.contains('药材科')]  # 数据量，(7048,12)
drug_tcms = drug_tcms.reset_index(drop=True)

print(drug_tcms.shape)  # (10864,9)
print(len(np.unique(drug_tcms['patient_id'])))  # 1079

# 合并自身免疫疾病病人的他克莫司用药和tdm检测数据
drug_test_tcms = pd.merge(drug_tcms,test_record_result_tdm, on=['patient_id'], how='inner')
# 删除start_datetime, test_datetime为空的数据。--> 没有
drug_test_tcms = drug_test_tcms[(drug_test_tcms['test_date'].notnull()) & (drug_test_tcms['start_datetime'].notnull())]
# 时间字段要注意数据类型，有些时间字段为str，有些为timestamp，类型冲突会报错
drug_test_tcms['start_datetime'] = drug_test_tcms['start_datetime'].apply(str_to_datatime)
drug_test_tcms['test_date'] = drug_test_tcms['test_date'].apply(str_to_datatime)

# end_datetime为空的数据赋值为start_datetime
aaa = drug_test_tcms[(drug_test_tcms['end_datetime'].astype('str') == 'nan')]
bbb = drug_test_tcms[drug_test_tcms['end_datetime'].astype('str') != 'nan']
aaa['end_datetime'] = aaa['start_datetime']
drug_test_tcms = pd.concat([aaa, bbb], axis=0)
drug_test_tcms = drug_test_tcms.reset_index(drop=True)

print(drug_test_tcms.shape)  # (4418,15)
print(len(np.unique(drug_test_tcms['patient_id'])))  # 152

writer = pd.ExcelWriter(project_path + '/result/df_1_自身免疫病人的他克莫司用药和tdm检测数据.xlsx')
drug_test_tcms.to_excel(writer)
writer.save()

# 补充他克莫司用药自身免疫病人的性别、年龄、身高、体重等数据
print('-----------------------------补充他克莫司用药自身免疫病人的性别、年龄、身高和体重-----------------------------------')
df_gender = df_gender[['patient_id', 'gender']].drop_duplicates(subset=['patient_id', 'gender'], keep='last')
df_age = df_age[['patient_id','age']].drop_duplicates(subset=['patient_id', 'age'], keep='last')
drug_test_tcms = pd.merge(drug_test_tcms,df_gender, on=['patient_id'],how='left')
drug_test_tcms = pd.merge(drug_test_tcms,df_age, on=['patient_id'], how='left')
# 合并身高、体重
df_figure = df_figure[['patient_id','身高(cm)','体重(kg)']]
df_figure = df_figure[df_figure['身高(cm)'].notnull()]
# 将病人的多个身高、体重数据取平均值
figure_list=[]
for i in np.unique(df_figure['patient_id']):
    temp=df_figure[df_figure['patient_id']==i]
    height = temp['身高(cm)'].astype('float').mean()
    temp['身高(cm)']= height
    weight = temp['体重(kg)'].astype('float').mean()
    temp['体重(kg)']= weight
    BMI=weight / (height * height) * 10000
    temp['BMI'] =BMI
    figure_list.append(temp)
df_figure_temp=figure_list[0]
for j in range(1,len(figure_list)):
    df_figure_temp=pd.concat([df_figure_temp,figure_list[j]],axis=0)
df_figure_temp=df_figure_temp.drop_duplicates(subset=['patient_id'],keep='last')
df_figure_temp=df_figure_temp.reset_index(drop=True)
drug_test_tcms = pd.merge(drug_test_tcms,df_figure_temp, on=['patient_id'],how='left')

# 从patient_vital文件中补充缺失的身高数据
df_patient_vital = patient_vital[patient_vital['sign_type']=='身高(cm)']
df_patient_vital['record_content'] = df_patient_vital['record_content'].apply(lambda x: x.replace('cm',''))
df_patient_vital['record_content'] = df_patient_vital['record_content'].astype('str').apply(lambda x: np.nan if x =='卧床'
                                                                                            else np.nan if x=='轮椅'
                                                                                            else np.nan if x=='抱入' else x)
temp_list=[]
for i in np.unique(df_patient_vital['patient_id']):
    temp=df_patient_vital[df_patient_vital['patient_id']==i]
    value=temp['record_content'].astype('float').mean()
    # print(value)
    temp['身高']=round(value,2)
    temp_list.append(temp)
height_sup=temp_list[0]
for j in range(1,len(temp_list)):
    height_sup=pd.concat([height_sup,temp_list[j]],axis=0)
height_sup = height_sup.drop_duplicates(subset=['patient_id'],keep='first')
height_sup=height_sup.reset_index(drop=True)

aaa=drug_test_tcms[drug_test_tcms['身高(cm)'].isnull()]
bbb=drug_test_tcms[drug_test_tcms['身高(cm)'].notnull()]
temp_list=[]
for i in np.unique(aaa['patient_id']):
    temp=aaa[aaa['patient_id']==i]
    # print(i)
    value = list(height_sup[height_sup['patient_id']==i]['身高'])[0]
    temp['身高(cm)']=value
    temp_list.append(temp)
temp_aaa=temp_list[0]
for j in range(1,len(temp_list)):
    temp_aaa=pd.concat([temp_aaa,temp_list[j]],axis=0)
drug_test_tcms=pd.concat([bbb,temp_aaa],axis=0)
# 男性为1，女性为0，性别数字化
drug_test_tcms['gender']=drug_test_tcms['gender'].astype('str').apply(lambda x: 0 if x=='女' else 1)
drug_test_tcms = drug_test_tcms.reset_index(drop=True)

print(drug_test_tcms.shape)  # (4418,20)
print(len(np.unique(drug_test_tcms['patient_id'])))  # 152

writer = pd.ExcelWriter(project_path + '/result/df_2_补充自身免疫病人的身高体重数据.xlsx')
drug_test_tcms.to_excel(writer)
writer.save()


# 需求2.2：tdm检测前7天内他克莫司用药筛选，注意他克莫司用药不能和tdm检测在同一天，一定要提前1天！！！
print('------------------tdm检测前7天内他克莫司用药筛选----------------------------------')
drug_test_tcms_7 = drug_test_tcms[(drug_test_tcms['test_date'] - datetime.timedelta(days=7) <= drug_test_tcms['start_datetime'])&
                                  (drug_test_tcms['start_datetime'] <= drug_test_tcms['test_date'] - datetime.timedelta(days=1))]
# 检测时间test_date升序，用药时间start_datetime降序，方便后面7-15天筛选。这样选出来是第一条tdm检测和最后一次用药。
drug_test_tcms_7 = drug_test_tcms_7.sort_values(by=['patient_id', 'test_date', 'start_datetime'],
                                                ascending=[True, True, False])
drug_test_tcms_7 = drug_test_tcms_7.reset_index()
del drug_test_tcms_7['index']

print(drug_test_tcms_7.shape)  # (391,20)
print(len(np.unique(drug_test_tcms_7['patient_id'])))  # 78

writer = pd.ExcelWriter(project_path + '/result/df_3_tdm检测前7天的他克莫司用药数据.xlsx')
drug_test_tcms_7.to_excel(writer)
writer.save()

# 他克莫司用药dosage处理。因为在两次tdm检测7-15天间隔判断之后，很多数据被删除，1/早、1/晚频次的dosage没法计算平均值。
print('------------------删除他克莫司用药dosage、frequency空值----------------------------------')

# 删除频次、剂型、剂量为空的样本
drug_test_tcms_7 = drug_test_tcms_7[(drug_test_tcms_7['drug_spec'].notnull()) & \
                                        (drug_test_tcms_7['dosage'].notnull()) & \
                                        (drug_test_tcms_7['frequency'].notnull())]
drug_test_tcms_7 = drug_test_tcms_7.reset_index()
del drug_test_tcms_7['index']

print(drug_test_tcms_7.shape)  # (255,20)
print(len(np.unique(drug_test_tcms_7['patient_id'])))  # 76

# 长嘱他克莫司用药dosage数据处理。
print('------------------他克莫司用药dosage数字化处理----------------------------------')
print(drug_test_tcms_7.columns)
# 1)'粒'根据剂型转换为mg数值
for i in range(drug_test_tcms_7.shape[0]):
    if '粒' in drug_test_tcms_7.loc[i, 'dosage']:
        num = drug_test_tcms_7.loc[i, 'dosage'].split('粒')[0]
        one_w = drug_test_tcms_7.loc[i, 'drug_spec'].split('mg')[0]
        drug_test_tcms_7.loc[i, 'dosage'] = str(float(one_w) * int(num))

# 2）dosage去除mg
drug_test_tcms_7['dosage'] = drug_test_tcms_7['dosage'].astype('str').apply(lambda x: x.replace('mg', ''))

# 长嘱他克莫司用药异常频次dosage数据处理。处理'1/早','1/晚'为每日两次，去两次用药的平均dosage，频次改为每日两次(自定义)
print('------------------他克莫司用药异常频次的dosage处理----------------------------------')
aaa = drug_test_tcms_7[
    (drug_test_tcms_7['frequency'].str.contains('1/早')) | (drug_test_tcms_7['frequency'].str.contains('1/晚'))]
bbb = drug_test_tcms_7[
    ~ (drug_test_tcms_7['frequency'].str.contains('1/早') | (drug_test_tcms_7['frequency'].str.contains('1/晚')))]
# 创建start_datetime_dosage，以时间为根据分组，计算平均用药dosage，频次改为每日两次(自定义)
aaa['start_datetime_dosage'] = aaa['start_datetime'].astype('str').apply(lambda x: x.split(' ')[0])

# 根据patient_id和start_datetime_dosage，计算平均用药dosage。
all_id = []
for i in np.unique(aaa['patient_id']):
    temp = aaa[aaa['patient_id'] == i]
    temp = temp.reset_index()
    del temp['index']
    # 计算patient_id下，某一天的早/天、晚/天频次的平均用药
    between_id = []
    for j in np.unique(temp['start_datetime_dosage']):
        temp_t = temp[temp['start_datetime_dosage'] == j]
        temp_t['dosage'] = temp_t['dosage'].astype('float').mean()
        between_id.append(temp_t)

    # 将某一patient_id的所有不同时间内的异常频次用药合并
    temp_between = between_id[0]
    for k in range(1, len(between_id)):
        temp_between = pd.concat([temp_between, between_id[k]], axis=0)  # 将groupby后的list类型转换成DataFrame
    temp_between = temp_between.reset_index()
    del temp_between['index']
    all_id.append(temp_between)

# 将所有patient_id下的异常频次用药合并
temp_all = all_id[0]
for k in range(1, len(all_id)):
    temp_all = pd.concat([temp_all, all_id[k]], axis=0)
temp_all['frequency'] = '每日两次(自定义)'
temp_all = temp_all.reset_index()
del temp_all['index']

drug_test_tcms_dosage = pd.concat([temp_all, bbb], axis=0)
drug_test_tcms_dosage = drug_test_tcms_dosage.sort_values(by=['patient_id', 'test_date', 'start_datetime'],
                                                            ascending=[True, True, False])
drug_test_tcms_dosage = drug_test_tcms_dosage.reset_index()
del drug_test_tcms_dosage['index']
del drug_test_tcms_dosage['start_datetime_dosage']

print(drug_test_tcms_dosage.shape)  # (255,20)
print(len(np.unique(drug_test_tcms_dosage['patient_id'])))  # 76

writer = pd.ExcelWriter(project_path + '/result/df_4_他克莫司dosage处理.xlsx')
drug_test_tcms_dosage.to_excel(writer)
writer.save()

# 2.2同一位病人两次TDM检测间隔时间主要分布在15天内，取第一次tdm检测；如果间隔在15天以上，可以认为每个样本相互独立。因为7天间隔的话，两次他克莫司用药间隔可能小于7天。
print('------------------同一位病人两次TDM检测间隔15天判断----------------------------------')
all_id = []
for i in np.unique(drug_test_tcms_dosage['patient_id']):
    temp = drug_test_tcms_dosage[drug_test_tcms_dosage['patient_id'] == i]
    temp = temp.reset_index()
    del temp['index']
    between_id = []
    j = 0
    while j < temp.shape[0]:
        # 取出符合要求的第一次tdm检测数据，其中他克莫司服药因为之前时间倒序排序，取得是最近的一次用药。
        between_id.append(temp.iloc[[j]])  # .iloc[[i]]取出dataframe；.loc[i]取出series
        k = j + 1
        # 同一个病人id的第j次tdm检测和第k次tdm检测，15天间隔
        while k < temp.shape[0]:
            # 两次tdm检测在15天内，只保留第一条。
            if temp.loc[j, 'test_date'] <= temp.loc[k, 'test_date'] - datetime.timedelta(days=15):
                k += 1
                continue
            else:
                break
        # 两次tdm检测间隔15天及以上，认为相互独立，break，将k赋值给j，下一次j循环会将k对应的tdm检测存入between_id
        j = k

    temp_between = between_id[0]
    for m in range(1, len(between_id)):
        temp_between = pd.concat([temp_between, between_id[m]], axis=0)  # list转换为DateFrame
    temp_between = temp_between.reset_index()
    del temp_between['index']
    all_id.append(temp_between)

drug_test_tcms_15 = all_id[0]
for n in range(1, len(all_id)):
    drug_test_tcms_15 = pd.concat([drug_test_tcms_15, all_id[n]], axis=0)
drug_test_tcms_15 = drug_test_tcms_15.reset_index()
del drug_test_tcms_15['index']

print(drug_test_tcms_15.shape)  # (228,20)
print(len(np.unique(drug_test_tcms_15['patient_id'])))  # 76

writer = pd.ExcelWriter(project_path + '/result/df_5_两次他克莫司检测间隔15天判断.xlsx')
drug_test_tcms_15.to_excel(writer)
writer.save()

# 需求3：提取距离他克莫司TDM检测最近的一次他克莫司长嘱(长期医嘱)的用药频次和剂量
print('---------------------3.最近的一次他克莫司长嘱用药的日剂量计算--------------------')
# 因为在上面drug_test_tcms_15中，检测时间test_date升序，用药时间start_datetime降序，取出的tdm检测对应的就是检测前最近的一次用药，666
# 在他克莫司用药医嘱筛选中drug_tcms，只提取的胶囊剂型！
drug_test_tcms_15 = drug_test_tcms_15[drug_test_tcms_15['drug_name'].str.contains('胶囊')]
print(drug_test_tcms_15.shape)  # (228,20)
print(len(np.unique(drug_test_tcms_15['patient_id'])))  # 76
# 筛选出长嘱用药
drug_test_tcms_long = drug_test_tcms_15[drug_test_tcms_15['long_d_order'] == 1]
drug_test_tcms_long = drug_test_tcms_long.reset_index()
del drug_test_tcms_long['index']

print(drug_test_tcms_long.shape)  # (145,15)
print(len(np.unique(drug_test_tcms_long['patient_id'])))  # 73

writer = pd.ExcelWriter(project_path + '/result/df_6_他克莫司长嘱用药.xlsx')
drug_test_tcms_long.to_excel(writer)
writer.save()

# 他克莫司用药dosage处理，挪到15天间隔过滤前。因为在tdm检测15天间隔判断之后，删除许多数据，1/早、1/晚没法具体处理。
# 他克莫司用药频次数字化处理
print('------------------他克莫司用药frequency处理----------------------------------')
print(np.unique(drug_test_tcms_long['frequency']))
one = ['1/单日', '1/日', '1/隔日', '1/午', 'ONCE']  # 周频次，改为0
two = ['1/12小时', '2/日', '2/日(餐前)', '每日两次(自定义)']
three = ['Tid']
drug_test_tcms_long['frequency'] = drug_test_tcms_long['frequency'].astype('str').apply(
    lambda x: 1 if x in one else 2 if x in two else 3 if x in three else 0)
drug_test_tcms_long['日剂量'] = drug_test_tcms_long['frequency'] * drug_test_tcms_long['dosage'].astype('float')

drug_test_tcms_frequency = drug_test_tcms_long[
    ['patient_id','long_d_order', 'drug_name', 'drug_spec', 'dosage', 'frequency', 'start_datetime', \
     'end_datetime', '日剂量','test_date', 'project_name', 'test_result', 'is_normal']]

drug_test_tcms_frequency['test_result'] = drug_test_tcms_frequency['test_result'].astype('str').apply(lambda x: re.sub(r'<|>', '',x))

print(drug_test_tcms_frequency.shape)  # (145,13)
print(len(np.unique(drug_test_tcms_frequency['patient_id'])))  # 73

writer = pd.ExcelWriter(project_path + '/result/df_7_他克莫司frequency处理.xlsx')
drug_test_tcms_frequency.to_excel(writer)
writer.save()

#  需求4：提取他克莫司TDM检测7天之内的最后的一次其他检验指标
#  4.1，先根据patient_id进行第一次数据分组，再根据tdm_time进行第二次数据分组，对DataFrame转换的数据进行时间排序，筛选tdm前7天内的其他检测。
#  4.2，按时间排序后，这样我们就可以根据patient_id、project_name进行最近一次的筛选，不同担心多个tdm检测前的多次相同其他检测指标被误删。
#  4.3，将具体的其他检测指标转换为列，将其整合到test_order_tcm_frequency建模数据中。
#  4.4，然后，我们再对其他指标进行处理：1)删除缺失值; 2) 不平衡指标, 正负样本数量相差悬殊; 3)人工选择删除无意义字段
#  4.5，然后，对其他检测指标进行数字化或分段处理，之后可以进行相关性分析，删除不相关指标。
#  其中，相关性分析: 分类变量(二分类，Mann-Whitney U test;多分类，方差分析-统计量F); 连续变量，pearson相关性检验(统计量r);
#  4.6，提取重要的指标，然后用随机森林进行缺失值插补。
print('---------------------4.提取他克莫司TDM检测7天之内的最后一次其他检验指标--------------------')
# print(np.unique(test_record_result['project_name']))
all_id = []
for i in np.unique(drug_test_tcms_frequency['patient_id']):
    # 根据patient_id进行第一次分类
    tdm_time = drug_test_tcms_frequency[drug_test_tcms_frequency['patient_id'] == i]  # 他克莫司的tdm检测数据
    # 检测时间排序
    tdm_time = tdm_time.sort_values(by=['test_date'], ascending=True)
    tdm_time = tdm_time.reset_index()
    del tdm_time['index']
    # 根据patient_id筛选出其他检测，并提取有效字段
    temp = test_record_result[test_record_result['patient_id'] == i]
    temp = temp[['patient_id', 'test_date', 'project_name', 'test_result', 'is_normal']]
    temp_other = temp[~temp['project_name'].str.contains('他克莫司')]  # 不包含他克莫司，为其他检验指标
    temp_other['test_date'] = temp_other['test_date'].apply(str_to_datatime)
    # 修改其他检测的字段名称，避免与tdm检测合并时发生字段名冲突
    temp_other = temp_other.rename(columns={'test_date': 'test_date_other', 'project_name': 'project_name_other',
                                            'test_result': 'test_result_other', 'is_normal': 'is_normal_other'})
    # 检测时间排序
    temp_other = temp_other.sort_values(by=['test_date_other'], ascending=True)
    temp_other = temp_other.reset_index()
    del temp_other['index']

    # 4.1，根据tdm_time进行第二次数据分组
    between_id = []
    for j in range(tdm_time.shape[0]):
        tdm_time_1 = tdm_time.iloc[[j]]
        time_1 = tdm_time.loc[j,'test_date']

        last_id = []
        for k in range(temp_other.shape[0]):
            # 筛选tdm前7天内的其他检测
            if time_1 - datetime.timedelta(days=7) <= temp_other.loc[k,'test_date_other'] <= time_1:
                last_id.append(temp_other.iloc[[k]])
        if last_id:
            temp_last = last_id[0]
            for m in range(1, len(last_id)):
                temp_last = pd.concat([temp_last, last_id[m]], axis=0)
            # 4.2，根据patient_id、project_name进行最近一次的筛选
            temp_last = temp_last.sort_values(by=['test_date_other'], ascending=True)
            temp_last = temp_last.drop_duplicates(subset=['patient_id', 'project_name_other'], keep='last')
            temp_last = temp_last.reset_index()
            del temp_last['index']
            # 4.3，将筛选出来的7天之内的最后一次其他检测的具体指标转换为列，整合到建模数据中
            for n in range(len(np.unique(temp_last['project_name_other']))):
                tdm_time_1[temp_last.loc[n, 'project_name_other']] = temp_last.loc[n, 'test_result_other']
            tdm_time_1 = tdm_time_1.reset_index()
            del tdm_time_1['index']
        between_id.append(tdm_time_1)

    # 将patient_id下所有符合要求的其他检测列数据合并。
    temp_between = between_id[0]
    for m in range(1, len(between_id)):
        temp_between = pd.concat([temp_between, between_id[m]], axis=0)
    temp_between = temp_between.reset_index()
    del temp_between['index']
    all_id.append(temp_between)

# 将所有patient_id的数据进行合并
tdm_7_other = all_id[0]
for n in range(1, len(all_id)):
    tdm_7_other = pd.concat([tdm_7_other, all_id[n]], axis=0)
tdm_7_other = tdm_7_other.reset_index()
del tdm_7_other['index']

print(tdm_7_other.shape)  # (145,484)
print(len(np.unique(tdm_7_other['patient_id'])))  # 73

writer = pd.ExcelWriter(project_path + '/result/df_8_提取tdm检测7天内最近的一次其他检测.xlsx')
tdm_7_other.to_excel(writer)
writer.save()


#  4.4，然后，我们再对其他指标进行处理：1)删除缺失值; 2) 不平衡指标, 正负样本数量相差悬殊; 3)人工选择删除无意义字段
print('---------------------他克莫司TDM检测7天内的其他检验指标预处理--------------------')
tdm_7_other_filter = tdm_7_other
# 4.4.1，删除缺失值超过50%的其他指标
for i in np.unique(tdm_7_other_filter.columns):
    other_up = tdm_7_other_filter[i].isnull().sum()
    other_down = tdm_7_other_filter[i].shape[0]
    if tdm_7_other_filter[i].isnull().sum()/tdm_7_other_filter[i].shape[0] >= 0.5:
        del tdm_7_other_filter[i]

print(tdm_7_other_filter.shape)  # (145,95)
print(len(np.unique(tdm_7_other_filter['patient_id'])))  # 73

# 4.4.2，删除分类严重不平衡的其他检测指标, 即分类变量中，如果某一变量占比超过90%，则认为该指标分类不平衡，删除
# 其中其他检测指标是从第14列开始的
column_list = list(tdm_7_other_filter.columns)[13:]
for i in column_list:
    # 提取指标数据
    df_temp = tdm_7_other_filter[tdm_7_other_filter[i].notnull()]
    df_temp = df_temp.reset_index()
    del df_temp['index']
    # 通过类别数目判断，指标是分类变量还是连续变量
    if len(np.unique(df_temp[i])) > 10:
        continue
    # 如果分类变量中某一变量的占比超过90%，则删除该指标
    num_1 = df_temp[i].value_counts()  # df一列中不同变量的数目
    num_2 = num_1.div(len(df_temp[i]))  # div除法，所有元素都除以相同数值
    num_3 = num_2.max()  # 取出最大值
    if num_3 >= 0.9:
        del tdm_7_other_filter[i]
    # print(num_1)
    # print(len(df_temp))

print(tdm_7_other_filter.shape)  # (145,90)
print(len(np.unique(tdm_7_other_filter['patient_id'])))  # 73

# 4.4.3，人工删除无意义的其他检测指标，试带法初筛变量也不要
del tdm_7_other_filter['Status']
del tdm_7_other_filter['国际标准比率']
del tdm_7_other_filter['透明度']
del tdm_7_other_filter['颜色']
del tdm_7_other_filter['电导率等级']
del tdm_7_other_filter['尿白细胞(试带法初筛)']
del tdm_7_other_filter['尿葡萄糖(试带法初筛)']
del tdm_7_other_filter['尿蛋白(试带法初筛)']
del tdm_7_other_filter['RBC.隐血(试带法初筛)']


print(tdm_7_other_filter.shape)  # (145,81)
print(len(np.unique(tdm_7_other_filter['patient_id'])))  # 73

writer = pd.ExcelWriter(project_path + '/result/df_9_tdm检测7天内最近的一次其他检测指标预处理.xlsx')
tdm_7_other_filter.to_excel(writer)
writer.save()

#  4.5，然后，对其他检测指标进行数字化或分段处理，之后可以进行相关性分析，删除不相关指标。
#  其中，相关性分析: 分类变量(二分类，Mann-Whitney U test;多分类，方差分析-统计量F); 连续变量，pearson相关性检验(统计量r);
print('---------------------他克莫司TDM检测7天内的其他检验指标的相关性检验--------------------')
'''
#  其中，分类变量(二分类，Mann-Whitney U test;多分类，方差分析-统计量F);
print('--------------------------多分类变量数字化---------------------------------')
# 试带法初筛变量，不要！！！
# 尿上皮细胞、尿葡萄糖、尿胆红素等分类变量，包含：阴性、阳性等定性变量。数字化转化，阴性用0表示，阳性用1表示.
discrete_list = ['尿白细胞(试带法初筛)','尿葡萄糖(试带法初筛)','尿蛋白(试带法初筛)','RBC.隐血(试带法初筛)']
df_discrete = tdm_7_other_filter[discrete_list]

for i in discrete_list:
    del tdm_7_other_filter[i]
    df_discrete[i] = df_discrete[i].astype('str').apply(lambda x: 0 if '阴性' in x else 1 if '阳性' in x else x)

print(df_discrete.shape)  # (3046,4)
print('--------------------------计算分类变量的Mann-Whitney U test---------------------------------')

from scipy import stats

u_list = []
p_list = []
q_list = []
column_list = list(df_discrete.columns)
for i in column_list:
    x= df_discrete[df_discrete[i].astype('float').notnull()][i]
    y= tdm_7_other_filter[df_discrete[i].astype('float').notnull()]['test_result']

    u, p = stats.mannwhitneyu(x,y)
    u = round(u, 2)
    p = round(p, 3)
    q = 'Mann-Whitney U test'
    print(i, u, p)

    u_list.append(u)
    p_list.append(p)
    q_list.append(q)
df_mann= pd.DataFrame(data={'离散检测指标': column_list,
                                'u值': u_list,
                                'p值': p_list,
                                '方法': q_list})
df_mann_1 = df_mann[df_mann['p值'] <= 0.05]
list_mann = list(df_mann_1.columns)
df_mann_2 = df_mann[df_mann['p值'] >= 0.05]
df_mann = pd.concat([df_mann_1,df_mann_2], axis=0)
df_mann = df_mann.reset_index()
del df_mann['index']

writer = pd.ExcelWriter(project_path + '/result/df_10_其他检测指标分类变量的Mann-Whitney U test.xlsx')
df_mann.to_excel(writer)
writer.save()

# 根据Mann-Whitnet U test过滤分类变量
df_discrete_filter = df_discrete[list(np.unique(df_mann['离散检测指标']))]
'''

#  连续变量，pearson相关性检验(统计量r);
print('--------------------------计算连续变量的spearson相关性系数---------------------------------')

from scipy import stats
t_list = []
p_list = []
q_list = []
column_list = list(tdm_7_other_filter.columns)[13:]
for i in column_list:
    # 删除连续变量中的<、>号
    tdm_7_other_filter[i] = tdm_7_other_filter[i].astype('str').apply(lambda x: re.sub(r'<|>', '',x))
    x= tdm_7_other_filter[tdm_7_other_filter[i].astype('float').notnull()][i]
    y= tdm_7_other_filter[tdm_7_other_filter[i].astype('float').notnull()]['test_result']
    t, p = stats.spearmanr(x,y)
    t = round(t, 2)
    p = round(p, 3)
    q = '斯皮尔曼'
    print(i, t, p)

    t_list.append(t)
    p_list.append(p)
    q_list.append(q)
df_spearmanr= pd.DataFrame(data={'连续检测指标': column_list,
                                't值': t_list,
                                'p值': p_list,
                                '方法': q_list})
df_spearmanr_1 = df_spearmanr[df_spearmanr['p值'] <= 0.05]
df_spearmanr_2 = df_spearmanr[df_spearmanr['p值'] >= 0.05]
df_spearmanr = pd.concat([df_spearmanr_1,df_spearmanr_2], axis=0)
df_spearmanr = df_spearmanr.reset_index()
del df_spearmanr['index']

writer = pd.ExcelWriter(project_path + '/result/df_10_其他检测指标连续变量的spearmanr相关性检测.xlsx')
df_spearmanr.to_excel(writer)
writer.save()

# 保存过滤后的分类变量和连续变量
print('--------------------------输出相关性检测过滤后的其他检测指标---------------------------------')

# 根据spearmanr相关性检测过滤连续变量
for i in list(np.unique(df_spearmanr_2['连续检测指标'])):
    del tdm_7_other_filter[i]

print(tdm_7_other_filter.shape)  # (145,23)
writer = pd.ExcelWriter(project_path + '/result/df_11_相关性检测过滤后的其他检测指标.xlsx')
tdm_7_other_filter.to_excel(writer)
writer.save()

#  4.6，提取重要的指标，用随机森林进行缺失值插补。
import missing_value_interpolation
print('------------------------用随机森林对重要的其他检测指标缺失值进行插补---------------------------------')

tdm_7_other_interpolation = missing_value_interpolation.missing_value_interpolation(tdm_7_other_filter)

print(tdm_7_other_interpolation.shape)
print(len(np.unique(tdm_7_other_interpolation['patient_id'])))

writer = pd.ExcelWriter(project_path + '/result/df_12_其他检测指标变量缺失值插补.xlsx')
tdm_7_other_interpolation.to_excel(writer)
writer.save()

# 需求5：提取他克莫司TDM检测前7天内的联合用药
#  5.1，先根据patient_id进行第一次数据分组，再根据tdm_time进行第二次数据分组，对DataFrame转换的数据进行时间排序，筛选tdm前7天内的其他用药。
#  5.2，按时间排序后，这样我们就可以根据patient_id、project_name进行最近一次的筛选，不同担心多次tdm检测前的多次相同其他用药被误删。
#  5.3，将具体的其他用药转换为列，将其整合到tdm_7_other_filter建模数据中。
#  5.4，然后，我们再对其他指标进行预处理：1)删除缺失值; 2) 不平衡指标, 正负样本数量相差悬殊; 3)人工选择删除无意义字段
#  5.5，然后，对其他用药进行数字化或分段处理，之后可以进行相关性分析，删除不相关指标。
#  其中，相关性分析: 分类变量(二分类，Mann-Whitney U test;多分类，方差分析-统计量F); 连续变量，pearson相关性检验(统计量r);
#  5.6，提取重要的其他用药，然后用随机森林进行缺失值插补。

# 糖皮质激素（地塞米松、甲泼尼龙、泼尼松、可的松）、质子泵抑制剂（奥美拉唑、泮托拉唑、艾普拉唑、雷贝拉唑、兰索拉唑、雷尼替丁）、
# 钙离子阻抗剂（硝苯地平、氨氯地平、尼群地平、非洛地平、地尔硫卓）、其他免疫抑制剂（环孢素、吗替麦考酚酯、环磷酰胺、硫唑嘌呤、甲氨蝶呤）、克拉霉素、阿奇霉素
















