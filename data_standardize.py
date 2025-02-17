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
print(data_raw.info())
# print(data_raw.sample(10))
data_raw.drop([
    'TNT1-Y太阳电池内板温度1', 'TNT14锂离子蓄电池组A温度1', 'TNT15锂离子蓄电池组A温度2',
    'TNT4-Y太阳电池内板温度4','INZ1_PCU输出母线备份电流','INZ1_VC1_PCU输出母线备份电流(VC1)','INZ6_VC1_-Y太阳电池阵电流（VC1）','INZ7_VC1_+Y太阳电池阵电流(VC1)','VNA2_VC1_A蓄电池整组电压(VC1)','VNA3_VC1_B蓄电池整组电压(VC1)','VNZ2_VC1MEA电压(S3R)(VC1)','VNZ3_VC1MEA电压(BCDR)(VC1)'],axis=1,inplace=True)
print('Train columns with null values:\n', data_raw.isnull().sum())
print("-" * 10)

print(data_raw.describe(include='all'))

data_new = data_raw.dropna()
print(data_new.info())
# data_new.to_csv('data/data_std.csv', encoding='utf-8')


fig1 = plt.figure(figsize=(15, 70), dpi=200)
for i, column in enumerate(data_new.columns):
    ax = fig1.add_subplot(len(data_new.columns), 1, i + 1)
    ax.plot(data_raw.loc[:, [column]], 'o', color='green', markersize=0.5)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))

    # plt.axes([0.14, 0.35, 0.77, 0.9])
    # ax.xaxis.set_tick_params(rotation=30)
    ax.set_title(column, FontProperties=font)
#plt.subplots_adjust(left=0.02, bottom=0.01, right=0.98, top=0.99, hspace=0.4, wspace=0.3)
# left=0.02, bottom=0.01, right=0.98, top=0.99 分别代表到画布的左侧和底部的距离占整幅图宽和高的比例
#plt.savefig('result/column_new.png')

# 对数据进行归一化、并且对数据'INA1_PCU输出母线电流'进行滑动平均去燥

INA1_PCU = data_new['INA1_PCU输出母线电流'].rolling(5).mean()
data_new.drop(['INA1_PCU输出母线电流'],axis=1,inplace=True)
satellite_data = pd.concat([data_new, INA1_PCU], axis=1).dropna()
satellite_np_data = satellite_data.as_matrix()
scaler = MinMaxScaler()
satellite_np_data = scaler.fit_transform(satellite_np_data)
print(satellite_np_data.shape)
index = satellite_data.index
columns = satellite_data.columns
data_rolling = pd.DataFrame(satellite_np_data, index=index, columns=columns)
# data_rolling.to_csv('data/data_rolling.csv', encoding='utf-8')

satellite_data_ = pd.DataFrame()
alt = pd.DataFrame()
for i in range(len(satellite_data)-5):
    a1 = satellite_data.iloc[i]
    a2 = satellite_data.iloc[i+1]
    a3 = satellite_data.iloc[i+2]
    a4 = satellite_data.iloc[i+3]
    a5 = satellite_data.iloc[i+4]
    d1=pd.DataFrame(a1).T
    d2=pd.DataFrame(a2).T
    d3=pd.DataFrame(a3).T
    d4=pd.DataFrame(a4).T
    d5=pd.DataFrame(a5).T
    satellite_data_=satellite_data_.append([d1])  #每行复制5倍
    satellite_data_=satellite_data_.append([d2])
    satellite_data_=satellite_data_.append([d3])
    satellite_data_=satellite_data_.append([d4])
    satellite_data_=satellite_data_.append([d5])

satellite_data_.to_csv('data/data_lstm.csv', encoding='utf-8')