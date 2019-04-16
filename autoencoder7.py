import numpy as np
np.random.seed(1337)  # for reproducibility
 
from keras.datasets import mnist
from keras.models import Model #泛型模型
from keras.layers import Dense, Input, Dropout
from keras import metrics,regularizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.constraints import maxnorm
from keras.layers.advanced_activations import LeakyReLU

##########################
# autoencoder6 去燥自编码器，分部训练
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
model_name = 'autoencoder7'
dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
satellite_data = pd.read_csv(
    'data/data_rolling.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)
# ['VNZ4A组蓄电池BEA信号','VNZ5B组蓄电池BEA信号','INZ6_-Y太阳电池阵电流','INZ7_+Y太阳电池阵电流','INZ14_ABCR1输入电流','INZ15_ABCR2输入电流']
# for column in satellite_data.columns:
#     if column not in ['INZ14_ABCR1输入电流','INZ15_ABCR2输入电流','INA4_A电池组充电电流','VNZ2MEA电压(S3R)',
#                       'VNZ3MEA电压(BCDR)','INA1_PCU输出母线电流','TNZ1PCU分流模块温度1','TNZ2PCU分流模块温度2',
#                       'TNZ6充放电模块温度2','TNZ7充放电模块温度3','VNA2_A蓄电池整组电压','VNA3_B蓄电池整组电压',
#                       'VNC1_蓄电池A单体1电压','VNC2蓄电池A单体2电压','VNC3蓄电池A单体3电压','VNC31蓄电池A电压',
#                       'VNC32蓄电池B电压','VNC4蓄电池A单体4电压','VNC5蓄电池A单体5电压','VNC6蓄电池A单体6电压',
#                       'VNC7蓄电池A单体7电压','VNC8蓄电池A单体8电压','VNC9蓄电池A单体9电压','INZ10_ABCR3输出电流',
#                       'INZ11_BBDR1输出电流','INZ12_BBDR2输出电流','INZ13_BBDR3输出电流','INZ8_ABDR1输出电流',
#                       'INZ9_ABDR2输出电流','INA2_A电池组放电电流']:
#         satellite_data[column] = 0.5

satellite_np_data = satellite_data.as_matrix()
print(satellite_np_data.shape)
index = satellite_data.index
columns = satellite_data.columns

index1 = satellite_data.iloc[0:80000].index
columns1 = satellite_data.iloc[0:80000].columns

x_train_ = satellite_np_data[0:80000]
x_test_ = satellite_np_data[80000:96700]
print(x_train_.shape)
print(x_test_.shape)
 
x_train, x_test = addNoise(x_train_,x_test_)

# data_target = pd.DataFrame(x_train, index=index1, columns=columns1)
# data_target.to_csv('result/{}/{}-noise.csv'.format(model_name,model_name), encoding='utf-8')

# this is our input placeholder
input_data = Input(shape=(34,))
 
# 编码层
#encoded = Dropout(0.2)(input_data)
encoded = Dense(18, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(input_data)
#encoded = Dropout(0.2)(encoded)
encoder_output = Dense(9, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(encoded)
 
# 解码层
decoded = Dense(9, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(encoder_output)
#decoded = Dropout(0.2)(decoded)
decoded = Dense(18, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(decoded)
#decoded = Dropout(0.2)(decoded)
decoded_output = Dense(34, activation='tanh')(decoded)
 
# 构建自编码模型
autoencoder = Model(inputs=input_data, outputs=decoded_output)
 
# 构建编码模型
encoder = Model(inputs=input_data, outputs=encoder_output)
 
# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mae', metrics=[metrics.mae])
print(autoencoder.summary())

# weight_file_path = 'model/{}/{}.h5'.format(model_name,'autoencoder7-weights1.100-0.00000012')
# autoencoder.load_weights(weight_file_path)

#weight_file_path = 'model/{}/{}.h5'.format(model_name,'autoencoder7-weights2.500-0.00000036')
#autoencoder.load_weights(weight_file_path)

# weight_file_path = 'model/{}/{}.h5'.format(model_name,'autoencoder7-weights3.100-0.00000061')
# autoencoder.load_weights(weight_file_path)

if DO_TRAINING:
    weight_file_path = 'model/{}/{}'.format(model_name,model_name)+'-weights2.{epoch:02d}-{val_loss:.8f}.h5'
    architecture_file_path = 'model/{}/{}-architecture.json'.format(model_name,model_name)
    open(architecture_file_path, 'w').write(autoencoder.to_json())
    # training
    checkpoint = ModelCheckpoint(weight_file_path)
    history = autoencoder.fit(x_train, x_train_, validation_data=(x_test,x_test_), callbacks=[checkpoint],epochs=500, batch_size=10, shuffle=True)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    plt.savefig('result/{}/loss.png'.format(model_name))
else:
    weight_file_path = 'model/{}/{}.h5'.format(model_name,'autoencoder7-weights7.70-0.00004884')
    autoencoder.load_weights(weight_file_path)

# features = encoder.predict(satellite_np_data)
# data_features = pd.DataFrame(features,index=index)
# data_features.to_csv('result/{}/{}-feat3-370.csv'.format(model_name,model_name), encoding='utf-8')

encoded_prd = autoencoder.predict(satellite_np_data)

data_target = pd.DataFrame(encoded_prd, index=index, columns=columns)
data_target.to_csv('result/{}/{}-prd7-70.csv'.format(model_name,model_name), encoding='utf-8')