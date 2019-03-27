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
mpl.rcParams['axes.titlesize'] = 6
mpl.rc('xtick', labelsize=6)  # 设置坐标轴刻度显示大小
mpl.rc('ytick', labelsize=6)
font_size = 1

# import data
# 列名字有中文的时候，encoding='utf-8',不然会出错
# index_col设置属性列，parse_dates设置是否解析拥有时间值的列
dateparser= lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
date_scaler = pd.read_csv(
    'data/data_scaler.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)
print(date_scaler.info())

dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data_rolling = pd.read_csv(
    'data/data_rolling.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

column='INA1_PCU输出母线电流'
fig1 = plt.figure(figsize=(10, 5))
ax = fig1.add_subplot(2, 1, 1)
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
ax.plot(date_scaler[0:80].loc[:, column], '-', color='green', linewidth=1.0,label='bus current')
ax.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
# ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time',fontsize=8,fontweight='bold')
plt.ylabel('Normalized current',fontsize=8,fontweight='bold')
plt.legend()
plt.grid(linestyle = "--")

ax = fig1.add_subplot(2, 1, 2)
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
ax.plot(data_rolling[0:80].loc[:, column], '-', color='green', linewidth=1.0,label='bus current')
ax.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
# ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time', fontsize=8,fontweight='bold')
plt.ylabel('Normalized current',fontsize=8,fontweight='bold')
plt.legend()
plt.grid(linestyle = "--")

plt.subplots_adjust(left=0.10, bottom=0.10, right=0.90, top=0.90, hspace=0.4, wspace=0.3)
# left=0.05, bottom=0.03, right=0.95, top=0.97 分别代表到画布的左侧和底部的距离占整幅图宽和高的比例
# plt.show()
plt.savefig('result/1.svg',format='svg')
