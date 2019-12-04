import numpy as np
np.random.seed(1337)  # for reproducibility
 
from keras.datasets import mnist
from keras.models import Sequential, Model #泛型模型
from keras.layers import Dense, Input, Dropout, LSTM, RepeatVector, TimeDistributed
from keras import metrics,regularizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.constraints import maxnorm
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model

##########################
# autoencoder9 LSTM自编码器
##########################


def addNoise(x_train, x_test):
    """
    add noise.
    :return:
    """
    NOISE_FACTOR = 0.02
    train_noisy = NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    test_noisy = NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = x_train + train_noisy
    x_test_noisy = x_test + test_noisy

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)     # limit into [0, 1]
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)   # limit into [0, 1]

    return x_train_noisy, x_test_noisy


DO_TRAINING = False
model_name = 'autoencoder9'
dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
satellite_data = pd.read_csv(
    'data/data_lstm.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

# dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
# satellite_data1 = pd.read_csv(
#     'data/data_rolling.csv',
#     sep=',',
#     index_col=0,
#     encoding='utf-8',
#     parse_dates=True,
#     date_parser=dateparser)

# dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
# satellite_data = pd.read_csv(
#     'data/data_anomaly_rolling2.csv',
#     sep=',',
#     index_col=0,
#     encoding='utf-8',
#     parse_dates=True,
#     date_parser=dateparser)

order = ['INA1_PCU输出母线电流','INZ14_ABCR1输入电流','INZ15_ABCR2输入电流',
'INA4_A电池组充电电流','INA2_A电池组放电电流', 'INZ10_ABCR3输出电流',
'INZ11_BBDR1输出电流','INZ12_BBDR2输出电流','INZ13_BBDR3输出电流',
'INZ8_ABDR1输出电流','INZ9_ABDR2输出电流','TNZ1PCU分流模块温度1',
'TNZ2PCU分流模块温度2', 'TNZ6充放电模块温度2','TNZ7充放电模块温度3',
'INZ6_-Y太阳电池阵电流','INZ7_+Y太阳电池阵电流', 'VNA2_A蓄电池整组电压',
'VNA3_B蓄电池整组电压', 'VNC1_蓄电池A单体1电压','VNC2蓄电池A单体2电压',
'VNC3蓄电池A单体3电压','VNC31蓄电池A电压','VNC32蓄电池B电压',
'VNC4蓄电池A单体4电压','VNC5蓄电池A单体5电压','VNC6蓄电池A单体6电压',
'VNC7蓄电池A单体7电压','VNC8蓄电池A单体8电压','VNC9蓄电池A单体9电压', 
'VNZ2MEA电压(S3R)','VNZ3MEA电压(BCDR)','VNZ4A组蓄电池BEA信号','VNZ5B组蓄电池BEA信号']
# satellite_data = satellite_data[order].iloc[:,0:32]

# ['INA1_PCU输出母线电流']
# ['INZ14_ABCR1输入电流','INZ15_ABCR2输入电流']
# ['INA4_A电池组充电电流']
# ['INA2_A电池组放电电流', 'INZ10_ABCR3输出电流','INZ11_BBDR1输出电流','INZ12_BBDR2输出电流','INZ13_BBDR3输出电流','INZ8_ABDR1输出电流','INZ9_ABDR2输出电流']
# ['TNZ1PCU分流模块温度1','TNZ2PCU分流模块温度2', 'TNZ6充放电模块温度2','TNZ7充放电模块温度3']
# ['INZ6_-Y太阳电池阵电流','INZ7_+Y太阳电池阵电流']
# ['VNA2_A蓄电池整组电压','VNA3_B蓄电池整组电压']
# ['VNC1_蓄电池A单体1电压','VNC2蓄电池A单体2电压','VNC3蓄电池A单体3电压','VNC31蓄电池A电压','VNC32蓄电池B电压','VNC4蓄电池A单体4电压','VNC5蓄电池A单体5电压','VNC6蓄电池A单体6电压','VNC7蓄电池A单体7电压','VNC8蓄电池A单体8电压','VNC9蓄电池A单体9电压']
# ['VNZ2MEA电压(S3R)','VNZ3MEA电压(BCDR)']
# ['VNZ4A组蓄电池BEA信号','VNZ5B组蓄电池BEA信号']

satellite_np_data = satellite_data.as_matrix()
print(satellite_np_data.shape)
index = satellite_data[0:483500].index
columns = satellite_data.columns

# index1 = satellite_data1.iloc[0:96700].index
# columns1 = satellite_data1.iloc[0:96700].columns

x_train_ = satellite_np_data[0:340000]
x_test_ = satellite_np_data[340000:483500]
print(x_train_.shape)
print(x_test_.shape)
 
x_train, x_test = addNoise(x_train_,x_test_)

time_window_size = 5
input_dim = x_train.shape[1]


satellite_np_data_ = np.reshape(
            satellite_np_data[0:483500],
            ((int)(satellite_np_data[0:483500].shape[0] / time_window_size),
             time_window_size, satellite_np_data[0:483500].shape[1]))


train_dataset = np.reshape(
            x_train,
            ((int)(x_train.shape[0] / time_window_size),
             time_window_size, x_train.shape[1]))

train_dataset_ = np.reshape(
            x_train_,
            ((int)(x_train_.shape[0] / time_window_size),
             time_window_size, x_train_.shape[1]))

test_dataset = np.reshape(
            x_test,
            ((int)(x_test.shape[0] / time_window_size),
             time_window_size, x_test.shape[1]))

test_dataset_ = np.reshape(
            x_test_,
            ((int)(x_test_.shape[0] / time_window_size),
             time_window_size, x_test_.shape[1]))

input_data = Input(batch_shape=(10, time_window_size, input_dim)) 
encoded = LSTM(units=100, activation='relu',return_sequences=True)(input_data)
decoded = LSTM(units=100, activation='relu', return_sequences=True)(encoded)
decoded = Dense(units = input_dim,activation='tanh')(decoded)
model = Model(inputs=input_data, outputs=decoded)

model.compile(optimizer='adam', loss='mse', metrics=[metrics.mse])
print(model.summary())


# model = Sequential()
# model.add(LSTM(100, activation='relu', input_shape=(time_window_size,input_dim)))
# model.add(RepeatVector(time_window_size))
# model.add(LSTM(100, activation='relu', return_sequences=True))
# model.add(TimeDistributed(Dense(input_dim)))
# model.compile(optimizer='adam', loss='mse')
# model.compile(optimizer='adam', loss='mse', metrics=[metrics.mse])
# print(model.summary())
# plot_model(model,to_file='result/{}/model.png'.format(model_name),show_shapes=True)

if DO_TRAINING:
    weight_file_path = 'model/{}/{}'.format(model_name,model_name)+'-weights.{epoch:02d}-{val_loss:.8f}.h5'
    architecture_file_path = 'model/{}/{}-architecture.json'.format(model_name,model_name)
    open(architecture_file_path, 'w').write(model.to_json())
    # training
    checkpoint = ModelCheckpoint(weight_file_path)

    history = model.fit(train_dataset,train_dataset_,
                    validation_data=(test_dataset,test_dataset_), callbacks=[checkpoint],epochs=2, batch_size=10, shuffle=False)


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    plt.savefig('result/{}/loss.png'.format(model_name))
else:
    weight_file_path = 'model/{}/{}.h5'.format(model_name,'autoencoder9-weights.499-0.00124347')
    model.load_weights(weight_file_path)

# features = encoder.predict(satellite_np_data)
# data_features = pd.DataFrame(features,index=index)
# data_features.to_csv('result/{}/{}-feat3-370.csv'.format(model_name,model_name), encoding='utf-8')

encoded_prd = model.predict(satellite_np_data_,batch_size=10)
encoded_prd_ = np.reshape(encoded_prd,(483500,34))
data_target = pd.DataFrame(encoded_prd_, index=index, columns=columns)
# data_target.to_csv('result/{}/{}-prd-499.csv'.format(model_name,model_name), encoding='utf-8')

data_target_ = pd.DataFrame()
a1 = data_target.iloc[0]
a2 = data_target.iloc[1]
a3 = data_target.iloc[2]
a4 = data_target.iloc[3]
a5 = data_target.iloc[4]
d1=pd.DataFrame(a1).T
d2=pd.DataFrame(a2).T
d3=pd.DataFrame(a3).T
d4=pd.DataFrame(a4).T
d5=pd.DataFrame(a5).T
data_target_=data_target_.append([d1])  #每行复制5倍
data_target_=data_target_.append([d2])
data_target_=data_target_.append([d3])
data_target_=data_target_.append([d4])
data_target_=data_target_.append([d5])
for i in range(1,96600):
    a = data_target.iloc[i*5]
    d=pd.DataFrame(a).T
    data_target_=data_target_.append([d])
data_target_.to_csv('result/{}/{}-prd-499-.csv'.format(model_name,model_name), encoding='utf-8')

# dataset_basic = data_target.as_matrix()
# # data_target.to_csv('result/{}/{}-ano7-249.csv'.format(model_name,model_name), encoding='utf-8')

# dist = np.linalg.norm(dataset_basic - satellite_np_data, axis=-1).reshape(-1,1)
# data_dist = pd.DataFrame(dist, index=index, columns=['norm'])
# data_dist.to_csv('result/{}/{}-ano7-249-2.csv'.format(model_name,model_name), encoding='utf-8')