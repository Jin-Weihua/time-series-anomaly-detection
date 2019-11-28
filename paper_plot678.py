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
dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
satellite_data = pd.read_csv(
    'data/data_rolling.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

dateparser= lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data_basic = pd.read_csv(
    'result/autoencoder2/autoencoder2-prd.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data_SAE = pd.read_csv(
    'result/autoencoder1/autoencoder1-prd.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)
# DAE和DEAS反过来展示，哎
dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data_DAE = pd.read_csv(
    'result/autoencoder6/autoencoder6-prd-180.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data_DAES = pd.read_csv(
    'result/autoencoder7/autoencoder7-prd7-249.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

# column1='VNZ4A组蓄电池BEA信号'
# ylabel = 'Normalized BEA value'
# column1='INZ14_ABCR1输入电流'
# ylabel = 'Normalized BCR input current'
column1='INA4_A电池组充电电流'
ylabel = 'Normalized battery set charge current'
fig1 = plt.figure(figsize=(8, 5),dpi=300)

ax1 = fig1.add_subplot(1, 4, 1)
ax1.spines['top'].set_visible(False)  #去掉上边框
ax1.spines['right'].set_visible(False) #去掉右边框
ax1.plot(data_basic[1000:7000].loc[:, column1], '-', color='green', linewidth=1.0)
# ax.plot(date_scaler[1000:7000].loc[:, column2], '-', color='C7', linewidth=1.0,label='BCR input current')
# ax.legend(loc='right',fontsize='xx-small')
# ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
ax1.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax1.xaxis.set_tick_params(rotation=30)
for label in ax1.get_xticklabels():
    label.set_visible(False)
for label in ax1.get_xticklabels()[::2]:
    label.set_visible(True)
# ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time(h)',fontsize=8,fontweight='bold')
plt.ylabel(ylabel,fontsize=8,fontweight='bold')
ax1.set_title('T-AE',fontsize=10)
# plt.legend()
plt.grid(linestyle = "--", alpha=0.5)

ax2 = fig1.add_subplot(1, 4, 2, sharey=ax1)
ax2.spines['top'].set_visible(False)  #去掉上边框
ax2.spines['right'].set_visible(False) #去掉右边框
ax2.spines['left'].set_visible(False) #去掉右边框
ax2.plot(data_SAE[1000:7000].loc[:, column1], '-', color='green', linewidth=1.0)
# ax.legend(loc='right',fontsize='xx-small')
ax2.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax2.xaxis.set_tick_params(rotation=30)
for label in ax2.get_xticklabels():
    label.set_visible(False)
for label in ax2.get_xticklabels()[::2]:
    label.set_visible(True)
for label in ax2.get_yticklabels():
    label.set_visible(False)
# ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time(h)', fontsize=8,fontweight='bold')
ax2.set_title('T-SAE',fontsize=10)
# plt.ylabel(ylabel,fontsize=8,fontweight='bold')
# plt.legend()
plt.grid(linestyle = "--", alpha=0.5)


ax3 = fig1.add_subplot(1, 4, 3, sharey=ax1)
ax3.spines['top'].set_visible(False)  #去掉上边框
ax3.spines['right'].set_visible(False) #去掉右边框
ax3.spines['left'].set_visible(False) #去掉右边
ax3.plot(data_DAE[1000:7000].loc[:, column1], '-', color='green', linewidth=1.0)
# ax.plot(date_scaler[1000:7000].loc[:, column3], '-', color='C7', linewidth=1.0,label='Solar cell array current')
# ax.legend(loc='right',fontsize='xx-small')
ax3.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax3.xaxis.set_tick_params(rotation=30)
for label in ax3.get_xticklabels():
    label.set_visible(False)
for label in ax3.get_xticklabels()[::2]:
    label.set_visible(True)
for label in ax3.get_yticklabels():
    label.set_visible(False)
    # ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time(h)', fontsize=8,fontweight='bold')
ax3.set_title('T-DAE',fontsize=10)
# plt.ylabel(ylabel,fontsize=8,fontweight='bold')
# plt.legend()
plt.grid(linestyle = "--", alpha=0.5)

ax4 = fig1.add_subplot(1, 4, 4, sharey=ax1)
ax4.spines['top'].set_visible(False)  #去掉上边框
ax4.spines['right'].set_visible(False) #去掉右边框
ax4.spines['left'].set_visible(False) #去掉右边框
ax4.plot(data_DAES[1000:7000].loc[:, column1], '-', color='green', linewidth=1.0)
# ax.legend(loc='right',fontsize='xx-small')
ax4.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax4.xaxis.set_tick_params(rotation=30)
for label in ax4.get_xticklabels():
    label.set_visible(False)
for label in ax4.get_xticklabels()[::2]:
    label.set_visible(True)
for label in ax4.get_yticklabels():
    label.set_visible(False)
# ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time(h)', fontsize=8,fontweight='bold')
ax4.set_title('ST-DAE',fontsize=10)
# plt.ylabel(ylabel,fontsize=8,fontweight='bold')
# plt.legend()
plt.grid(linestyle = "--", alpha=0.5)

fig2 = plt.figure(figsize=(5, 5),dpi=300)

ax5 = fig2.add_subplot(2, 2, 1)
ax5.spines['top'].set_visible(False)  #去掉上边框
ax5.spines['right'].set_visible(False) #去掉右边框
ax5.spines['left'].set_visible(False) #去掉右边框
ax5.plot(data_basic[1000:7000].loc[:, column1]-satellite_data[1000:7000].loc[:, column1], '-', color='#2a5caa', linewidth=1.0, label='T-AE')
ax5.plot(data_SAE[1000:7000].loc[:, column1] -satellite_data[1000:7000].loc[:, column1],'-', color='#87843b', linewidth=1.0,label='T-SAE')
ax5.plot(data_DAE[1000:7000].loc[:, column1] -satellite_data[1000:7000].loc[:, column1],'-', color='#faa755', linewidth=1.0,label='T-DAE')
ax5.plot(data_DAES[1000:7000].loc[:, column1] -satellite_data[1000:7000].loc[:, column1],'-', color='#905d1d', linewidth=1.0,label='ST-DAE')
# ax.legend(loc='right',fontsize='xx-small')
ax5.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax5.xaxis.set_tick_params(rotation=30)
for label in ax5.get_xticklabels():
    label.set_visible(False)
for label in ax5.get_xticklabels()[::2]:
    label.set_visible(True)
for label in ax5.get_yticklabels():
    label.set_visible(False)
# ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time(h)', fontsize=8,fontweight='bold')
ax5.set_title('ST-DAE',fontsize=10)
# plt.ylabel(ylabel,fontsize=8,fontweight='bold')
# plt.legend()
plt.grid(linestyle = "--", alpha=0.5)


plt.subplots_adjust(left=0.10, bottom=0.15, right=0.95, top=0.90, hspace=0.4, wspace=0.1)
# left=0.05, bottom=0.03, right=0.95, top=0.97 分别代表到画布的左侧和底部的距离占整幅图宽和高的比例
plt.show()
plt.savefig('result/8.eps',format='eps')
plt.savefig('result/8.tiff',format='tiff')