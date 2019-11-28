#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_standardize.py
@Time    :   2019/01/17 19:58:52
@Author  :   靳卫华 
@Version :   1.0
@Contact :   wh.jin@hotmail.com
@Desc    :   None
'''

# access to system parameters
import sys
print("Python version: {}".format(sys.version))

# collection of functions for data processing and analysis
# modeled after R dataframes with SQL like features
import pandas as pd
print("pandas version: {}".format(pd.__version__))

# collection of functions for scientific and publication-ready visualization
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.font_manager import FontProperties
print("matplotlib version: {}".format(matplotlib.__version__))

# foundational package for scientific computing
import numpy as np
print("NumPy version: {}".format(np.__version__))
from sklearn.preprocessing import MinMaxScaler
# ignore warnings
# import warnings
# warnings.filterwarnings('ignore')
print('-' * 25)

font = FontProperties(fname='/Library/Fonts/Songti.ttc')
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
mpl.rcParams['axes.titlesize'] = 7
mpl.rc('xtick', labelsize=5)  # 设置坐标轴刻度显示大小
mpl.rc('ytick', labelsize=5)
font_size = 5

# import data
# 列名字有中文的时候，encoding='utf-8',不然会出错
# index_col设置属性列，parse_dates设置是否解析拥有时间值的列
dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data_raw = pd.read_csv(
    'data/data.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

data_raw.drop([
    'TNT1-Y太阳电池内板温度1', 'TNT14锂离子蓄电池组A温度1', 'TNT15锂离子蓄电池组A温度2',
    'TNT4-Y太阳电池内板温度4','INZ1_PCU输出母线备份电流','INZ1_VC1_PCU输出母线备份电流(VC1)','INZ6_VC1_-Y太阳电池阵电流（VC1）','INZ7_VC1_+Y太阳电池阵电流(VC1)','VNA2_VC1_A蓄电池整组电压(VC1)','VNA3_VC1_B蓄电池整组电压(VC1)','VNZ2_VC1MEA电压(S3R)(VC1)','VNZ3_VC1MEA电压(BCDR)(VC1)'],axis=1,inplace=True)

data_new = data_raw.dropna()
print(data_new.info())

# 对数据进行归一化、并且对数据'INA1_PCU输出母线电流'进行滑动平均去燥

INA1_PCU = data_new['INA1_PCU输出母线电流'].rolling(5).mean()
data_new.drop(['INA1_PCU输出母线电流'],axis=1,inplace=True)
satellite_data = pd.concat([data_new, INA1_PCU], axis=1).dropna()

['INZ11_BBDR1输出电流','INZ12_BBDR2输出电流','INZ13_BBDR3输出电流','INA2_A电池组放电电流','VNZ2MEA电压(S3R)']

satellite_np_data = satellite_data.as_matrix()

scaler = MinMaxScaler()
satellite_np_data = scaler.fit_transform(satellite_np_data)

# data_1 = satellite_data.loc['2016-11-25  20:38:52':'2016-11-25  21:12:04']
data_1 = satellite_data.iloc[2650:3655]
i = 0

# # 异常一
# for index,row in satellite_data.iterrows():
#     if i > 2650 and i < 3655:
#         satellite_data.iloc[i]['INZ11_BBDR1输出电流'] = row['INZ11_BBDR1输出电流'] + row['INZ13_BBDR3输出电流']/2.0
#         satellite_data.iloc[i]['INZ12_BBDR2输出电流'] = row['INZ12_BBDR2输出电流'] + row['INZ13_BBDR3输出电流']/2.0
#         satellite_data.iloc[i]['INZ13_BBDR3输出电流'] = 0
#     i = i + 1

# 异常二
for index,row in satellite_data.iterrows():
    if i > 2670 and i < 3630:
        satellite_data.iloc[i]['INA2_A电池组放电电流'] = row['INA2_A电池组放电电流'] - 10
        satellite_data.iloc[i]['VNZ2MEA电压(S3R)'] = row['VNZ2MEA电压(S3R)'] + 5
    i = i + 1

data_2 = satellite_data.iloc[2650:3655]
data_new_anomaly = satellite_data

INA1_PCU = data_new_anomaly['INA1_PCU输出母线电流'].rolling(5).mean()
data_new_anomaly.drop(['INA1_PCU输出母线电流'],axis=1,inplace=True)
satellite_data_anomaly = pd.concat([data_new_anomaly, INA1_PCU], axis=1).dropna()
satellite_np_data_anomaly = satellite_data_anomaly.as_matrix()

satellite_np_data_anomaly = scaler.transform(satellite_np_data_anomaly)

print(satellite_np_data_anomaly.shape)
index = satellite_data_anomaly.index
columns = satellite_data_anomaly.columns
data_rolling = pd.DataFrame(satellite_np_data_anomaly, index=index, columns=columns)
data_rolling.to_csv('data/data_anomaly_rolling2.csv', encoding='utf-8')