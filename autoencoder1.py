import numpy as np
np.random.seed(1337)  # for reproducibility
 
from keras.datasets import mnist
from keras.models import Model #泛型模型
from keras.layers import Dense, Input
from keras import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

##########################
# 只使用全连接Dense
##########################
model_name = 'autoencoder1'
dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
satellite_data = pd.read_csv(
    'data/data_rolling.csv.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)
# column = ['INA1_PCU输出母线电流','INA4_A电池组充电电流','INA2_A电池组放电电流','TNZ1PCU分流模块温度1','INZ6_-Y太阳电池阵电流','VNA2_A蓄电池整组电压','VNC1_蓄电池A单体1电压','VNZ2MEA电压(S3R)','VNZ4A组蓄电池BEA信号']
satellite_np_data = satellite_data.as_matrix()
print(satellite_np_data.shape)
index = satellite_data.index
columns = satellite_data.columns
# input_dataset = np.reshape(
#         satellite_np_data,
#         ((int)(satellite_np_data.shape[0] / time_window_size),
#             time_window_size, satellite_np_data.shape[1]))

x_train = satellite_np_data[0:80000]
x_test = satellite_np_data[80000:96700]
print(x_train.shape)
print(x_test.shape)
 
# this is our input placeholder
input_data = Input(shape=(34,))
 
# 编码层
encoded = Dense(18, activation='selu')(input_data)
encoder_output = Dense(9, activation='selu')(encoded)
 
# 解码层
decoded = Dense(9, activation='selu')(encoder_output)
decoded = Dense(18, activation='selu')(decoded)
decoded_output = Dense(34, activation='selu')(encoded)
 
# 构建自编码模型
autoencoder = Model(inputs=input_data, outputs=decoded_output)
 
# 构建编码模型
# encoder = Model(inputs=input_data, outputs=encoder_output)
 
# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mae', metrics=[metrics.mae])
print(autoencoder.summary())
 
# training
history = autoencoder.fit(x_train, x_train,validation_data=(x_test,x_test), epochs=200, batch_size=10, shuffle=True)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
plt.savefig('result/{}/loss.png'.format(model_name))

# plotting
encoded_prd = autoencoder.predict(x_train)

data_target = pd.DataFrame(x_train, index=index, columns=columns)
data_target.to_csv('data/x_train.csv', encoding='utf-8')

data_target = pd.DataFrame(encoded_prd, index=index, columns=columns)
data_target.to_csv('data/test.csv', encoding='utf-8')