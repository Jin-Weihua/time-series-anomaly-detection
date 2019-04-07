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
    'data/data_rolling.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)
print(date_scaler.info())

dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data_rolling = pd.read_csv(
    'result/autoencoder2/autoencoder2-prd.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

column1='VNZ4A组蓄电池BEA信号'
column2='INZ14_ABCR1输入电流'
column3='INZ6_-Y太阳电池阵电流'
column4='INA4_A电池组充电电流'
fig1 = plt.figure(figsize=(12, 5))

ax = fig1.add_subplot(1, 6, 1)
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
ax.plot(date_scaler[1000:7000].loc[:, column1], '-', color='green', linewidth=1.0,label='BEA')
ax.plot(date_scaler[1000:7000].loc[:, column2], '-', color='C7', linewidth=1.0,label='BCR input current')
ax.legend(loc='right',fontsize='xx-small')
# ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
ax.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax.xaxis.set_tick_params(rotation=30)
for label in ax.get_xticklabels():
    label.set_visible(False)
for label in ax.get_xticklabels()[::2]:
    label.set_visible(True)
# ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time(h)',fontsize=8,fontweight='bold')
plt.ylabel('Normalized value',fontsize=8,fontweight='bold')
# plt.legend()
plt.grid(linestyle = "--", alpha=0.4)

ax = fig1.add_subplot(1, 6, 2)
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
ax.plot(data_rolling[1000:7000].loc[:, column1], '-', color='green', linewidth=1.0,label='BEA')
ax.legend(loc='right',fontsize='xx-small')
ax.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax.xaxis.set_tick_params(rotation=30)
for label in ax.get_xticklabels():
    label.set_visible(False)
for label in ax.get_xticklabels()[::2]:
    label.set_visible(True)
# ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time(h)', fontsize=8,fontweight='bold')
plt.ylabel('Normalized value',fontsize=8,fontweight='bold')
# plt.legend()
plt.grid(linestyle = "--", alpha=0.4)


ax = fig1.add_subplot(1, 6, 3)
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
ax.plot(date_scaler[1000:7000].loc[:, column2], '-', color='green', linewidth=1.0,label='BCR input current')
ax.plot(date_scaler[1000:7000].loc[:, column3], '-', color='C7', linewidth=1.0,label='Solar Cell Array Current')
ax.legend(loc='right',fontsize='xx-small')
ax.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax.xaxis.set_tick_params(rotation=30)
for label in ax.get_xticklabels():
    label.set_visible(False)
for label in ax.get_xticklabels()[::2]:
    label.set_visible(True)
    # ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time(h)', fontsize=8,fontweight='bold')
plt.ylabel('Normalized value',fontsize=8,fontweight='bold')
# plt.legend()
plt.grid(linestyle = "--", alpha=0.4)

ax = fig1.add_subplot(1, 6, 4)
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
ax.plot(data_rolling[1000:7000].loc[:, column2], '-', color='green', linewidth=1.0,label='BCR input current')
ax.legend(loc='right',fontsize='xx-small')
ax.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax.xaxis.set_tick_params(rotation=30)
for label in ax.get_xticklabels():
    label.set_visible(False)
for label in ax.get_xticklabels()[::2]:
    label.set_visible(True)
# ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time(h)', fontsize=8,fontweight='bold')
plt.ylabel('Normalized value',fontsize=8,fontweight='bold')
# plt.legend()
plt.grid(linestyle = "--", alpha=0.4)


ax = fig1.add_subplot(1, 6, 5)
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
ax.plot(date_scaler[1000:7000].loc[:, column1], '-', color='green', linewidth=1.0,label='BEA')
ax.plot(date_scaler[1000:7000].loc[:, column4], '-', color='C7', linewidth=1.0,label='Solar Cell Array Current')
ax.legend(loc='right',fontsize='xx-small')
ax.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax.xaxis.set_tick_params(rotation=30)
for label in ax.get_xticklabels():
    label.set_visible(False)
for label in ax.get_xticklabels()[::2]:
    label.set_visible(True)
    # ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time(h)', fontsize=8,fontweight='bold')
plt.ylabel('Normalized value',fontsize=8,fontweight='bold')
# plt.legend()
plt.grid(linestyle = "--", alpha=0.4)

ax = fig1.add_subplot(1, 6, 6)
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
ax.plot(data_rolling[1000:7000].loc[:, column4], '-', color='green', linewidth=1.0,label='BCR input current')
ax.legend(loc='right',fontsize='xx-small')
ax.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax.xaxis.set_tick_params(rotation=30)
for label in ax.get_xticklabels():
    label.set_visible(False)
for label in ax.get_xticklabels()[::2]:
    label.set_visible(True)
# ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time(h)', fontsize=8,fontweight='bold')
plt.ylabel('Normalized value',fontsize=8,fontweight='bold')
# plt.legend()
plt.grid(linestyle = "--", alpha=0.4)


plt.subplots_adjust(left=0.10, bottom=0.15, right=0.95, top=0.90, hspace=0.4, wspace=0.4)
# left=0.05, bottom=0.03, right=0.95, top=0.97 分别代表到画布的左侧和底部的距离占整幅图宽和高的比例
# plt.show()
plt.savefig('result/2.svg',format='svg')
