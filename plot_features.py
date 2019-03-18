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

model_name = 'autoencoder4'
dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data_raw1 = pd.read_csv(
    'result/{}/{}-feat3-370.csv'.format(model_name,model_name),
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)
columns = ['0','1','2','3','4','5','6','7','8']
data_raw = data_raw1.loc[:,columns].iloc[0:10000]

data = []
columns = ['0','1','2','3','4','5','6','7','8']
y = data_raw[columns]
data.append(go.Scatter(x=data_raw.index, y=data_raw['6'], mode='markers+lines',name='features'))

plotly.offline.plot(data, filename='result/{}/feat0.html'.format(model_name), auto_open=True)
