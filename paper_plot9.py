#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   plot.py
@Time    :   2018/12/08 18:31:57
@Author  :   靳卫华 
@Version :   1.0
@Contact :   wh.jin@hotmail.com
@Desc    :   None
'''

# here put the import lib
import plotly
plotly.tools.set_credentials_file(username='wh.jin', api_key='DADs4nY8rboe9LuL6Iuz')
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
print("pandas version: {}".format(pd.__version__))

# collection of functions for scientific and publication-ready visualization
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.font_manager import FontProperties
print("matplotlib version: {}".format(matplotlib.__version__))

font = FontProperties(fname='/Library/Fonts/Songti.ttc')
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
mpl.rcParams['axes.titlesize'] = 6
mpl.rc('xtick', labelsize=6)  # 设置坐标轴刻度显示大小
mpl.rc('ytick', labelsize=6)
font_size = 1

# foundational package for scientific computing
import numpy as np

dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data_raw = pd.read_csv(
    'data/data_rolling.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data_raw1 = pd.read_csv(
    'data/data_anomaly_rolling.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

data_prd1 = pd.read_csv(
    'result/autoencoder7/autoencoder7-ano7-249.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

columns = ['INZ11_BBDR1输出电流','INZ12_BBDR2输出电流','INZ13_BBDR3输出电流']
column_norm = ['norm']

ylabel = 'Normalized value'

column1 = 'INZ11_BBDR1输出电流'
column2 = 'INZ12_BBDR2输出电流'
column3 = 'INZ13_BBDR3输出电流'
column4 = 'norm'

fig1 = plt.figure(figsize=(8, 5),dpi=300)


ax = fig1.add_subplot(1, 4, 1)
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
ax.plot(data_raw[1500:4500].loc[:, column1], '-', color='green', linewidth=1.0,label='BDR1 output current')
ax.plot(data_raw1[1500:4500].loc[:, column1], '-', color='C7', linewidth=1.0,label='BDR1 output current(A)')

ax.legend(loc='right',fontsize='xx-small')
# ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
ax.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax.xaxis.set_tick_params(rotation=30)
for label in ax.get_xticklabels():
    label.set_visible(False)
for label in ax.get_xticklabels()[::2]:
    label.set_visible(True)
ax.set_title('a',fontsize=10)
# ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time(h)',fontsize=8,fontweight='bold')
plt.ylabel('Normalized value',fontsize=8,fontweight='bold')
# plt.legend()
plt.grid(linestyle = "--", alpha=0.4)


ax1 = fig1.add_subplot(1, 4, 2, sharey=ax)
ax1.spines['top'].set_visible(False)  #去掉上边框
ax1.spines['right'].set_visible(False) #去掉右边框
ax1.spines['left'].set_visible(False) #去掉左边框
ax1.plot(data_raw[1500:4500].loc[:, column2], '-', color='green', linewidth=1.0,label='BDR2 output current')
ax1.plot(data_raw1[1500:4500].loc[:, column2], '-', color='C7', linewidth=1.0,label='BDR2 output current(A)')

ax1.legend(loc='right',fontsize='xx-small')
# ax1.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax1.xaxis.set_tick_params(rotation=30)
for label in ax1.get_xticklabels():
    label.set_visible(False)
for label in ax1.get_xticklabels()[::2]:
    label.set_visible(True)
for label in ax1.get_yticklabels():
    label.set_visible(False)
ax1.set_title('b',fontsize=10)
# ax.set_title('bus current',fontsize=12,fontweight='bold')
plt.xlabel('Time(h)',fontsize=8,fontweight='bold')
# plt.ylabel(ylabel,fontsize=8,fontweight='bold')
# ax1.set_title('Basic-T',fontsize=10)
# plt.legend()
plt.grid(linestyle = "--", alpha=0.4)


ax1 = fig1.add_subplot(1, 4, 3, sharey=ax)
ax1.spines['top'].set_visible(False)  #去掉上边框
ax1.spines['right'].set_visible(False) #去掉右边框
ax1.spines['left'].set_visible(False) #去掉左边框
ax1.plot(data_raw[1500:4500].loc[:, column3], '-', color='green', linewidth=1.0,label='BDR3 output current')
ax1.plot(data_raw1[1500:4500].loc[:, column3], '-', color='C7', linewidth=1.0,label='BDR3 output current(A)')

ax1.legend(loc='right',fontsize='xx-small')
# ax1.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax1.xaxis.set_tick_params(rotation=30)
for label in ax1.get_xticklabels():
    label.set_visible(False)
for label in ax1.get_xticklabels()[::2]:
    label.set_visible(True)
for label in ax1.get_yticklabels():
    label.set_visible(False)
# ax.set_title('bus current',fontsize=12,fontweight='bold')
ax1.set_title('c',fontsize=10)
plt.xlabel('Time(h)',fontsize=8,fontweight='bold')
# plt.ylabel(ylabel,fontsize=8,fontweight='bold')
# ax1.set_title('Basic-T',fontsize=10)
# plt.legend()
plt.grid(linestyle = "--", alpha=0.5)

ax3 = fig1.add_subplot(1, 4, 4, sharey=ax)
ax3.spines['top'].set_visible(False)  #去掉上边框
ax3.spines['right'].set_visible(False) #去掉右边框
ax3.spines['left'].set_visible(False) #去掉右边
ax3.plot(data_prd1[1500:4500].loc[:, column4], '-', color='red', linewidth=1.0,label='anomaly score')
ax3.legend(loc='right',fontsize='xx-small')
# ax3.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
ax3.xaxis.set_tick_params(rotation=30)
for label in ax3.get_xticklabels():
    label.set_visible(False)
for label in ax3.get_xticklabels()[::2]:
    label.set_visible(True)
for label in ax3.get_yticklabels():
    label.set_visible(False)
ax3.set_title('d',fontsize=10)
plt.xlabel('Time(h)', fontsize=8,fontweight='bold')
# ax3.set_title('DAE-T',fontsize=10)
# plt.ylabel(ylabel,fontsize=8,fontweight='bold')
# plt.legend()
plt.grid(linestyle = "--", alpha=0.5)

plt.subplots_adjust(left=0.10, bottom=0.15, right=0.95, top=0.90, hspace=0.4, wspace=0.1)
# left=0.05, bottom=0.03, right=0.95, top=0.97 分别代表到画布的左侧和底部的距离占整幅图宽和高的比例
# plt.show()
plt.savefig('result/9.eps',format='eps')
plt.savefig('result/9.tiff',format='tiff')


# data = []
# data_raw = data_raw.loc[:,columns].iloc[0:10000]
# data_prd = data_raw1.loc[:,columns].iloc[0:10000]
# data_norm = data_prd1.loc[:,column_norm].iloc[0:10000]
# column = 'INZ11_BBDR1输出电流' 
# data.append(go.Scatter(x=data_raw.index, y=data_raw[column], mode='markers+lines', name=column))
# data.append(go.Scatter(x=data_prd.index, y=data_prd[column], mode='markers', name=column))
# column = 'INZ12_BBDR2输出电流' 
# data.append(go.Scatter(x=data_raw.index, y=data_raw[column], mode='markers+lines', name=column))
# data.append(go.Scatter(x=data_prd.index, y=data_prd[column], mode='markers', name=column))
# column = 'INZ13_BBDR3输出电流' 
# data.append(go.Scatter(x=data_raw.index, y=data_raw[column], mode='markers+lines', name=column))
# data.append(go.Scatter(x=data_prd.index, y=data_prd[column], mode='markers', name=column))
# # column = 'norm'
# # data.append(go.Scatter(x=data_prd.index, y=data_prd[column], mode='markers', name=column))



# plotly.offline.plot(data, filename='result/9.html', auto_open=True)
