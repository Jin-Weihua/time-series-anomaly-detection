import numpy as np
np.random.seed(1337)  # for reproducibility
 
from keras.datasets import mnist
from keras.models import Model #泛型模型
from keras.layers import Dense, Input, Dropout, Concatenate, concatenate
from keras import metrics,regularizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.constraints import maxnorm
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model

##########################
# autoencoder8 多输入自编码器
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
model_name = 'autoencoder8'
dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
satellite_data = pd.read_csv(
    'data/data_rolling.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

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
satellite_data = satellite_data[order]#.iloc[:,0:34]
# ['INA1_PCU输出母线电流']
# ['INZ14_ABCR1输入电流','INZ15_ABCR2输入电流','INA4_A电池组充电电流']
# ['INA2_A电池组放电电流', 'INZ10_ABCR3输出电流','INZ11_BBDR1输出电流','INZ12_BBDR2输出电流','INZ13_BBDR3输出电流','INZ8_ABDR1输出电流','INZ9_ABDR2输出电流']
# ['TNZ1PCU分流模块温度1','TNZ2PCU分流模块温度2', 'TNZ6充放电模块温度2','TNZ7充放电模块温度3']
# ['INZ6_-Y太阳电池阵电流','INZ7_+Y太阳电池阵电流']
# ['VNA2_A蓄电池整组电压','VNA3_B蓄电池整组电压']
# ['VNC1_蓄电池A单体1电压','VNC2蓄电池A单体2电压','VNC3蓄电池A单体3电压','VNC31蓄电池A电压','VNC32蓄电池B电压','VNC4蓄电池A单体4电压','VNC5蓄电池A单体5电压','VNC6蓄电池A单体6电压','VNC7蓄电池A单体7电压','VNC8蓄电池A单体8电压','VNC9蓄电池A单体9电压']
# ['VNZ2MEA电压(S3R)','VNZ3MEA电压(BCDR)']
# ['VNZ4A组蓄电池BEA信号','VNZ5B组蓄电池BEA信号']

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
index = satellite_data.index
columns = satellite_data.iloc[:,[0,1,2,11,12,13,14,17,18,19,20,21,22,23,24,25,26,27,28,29]].columns

index1 = satellite_data.iloc[0:80000].index
columns1 = satellite_data.iloc[0:80000].columns

x_train_ = satellite_np_data[0:80000]
x_test_ = satellite_np_data[80000:96700]
print(x_train_.shape)
print(x_test_.shape)
 
x_train, x_test = addNoise(x_train_,x_test_)

first_input = Input(shape=(1, ))
first_dense = Dense(1, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(first_input)

second_input = Input(shape=(2, ))
second_dense = Dense(1, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(second_input)

# second_input1 = Input(shape=(1, ))
# second_dense1 = Dense(1, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(second_input1)

# third_input_1 = Input(shape=(1, ))
# third_dense_1 = Dense(1, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(third_input_1)

# third_input = Input(shape=(6, ))
# third_dense = Dense(1, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(third_input)

fouth_input = Input(shape=(4, ))
fouth_dense = Dense(1, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(fouth_input)

# fifth_input = Input(shape=(2, ))
# fifth_dense = Dense(1, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(fifth_input)

sixth_input = Input(shape=(2, ))
sixth_dense = Dense(1, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(sixth_input)

seventh_input = Input(shape=(11, ))
seventh_dense = Dense(1, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(seventh_input)

# eighth_input = Input(shape=(2, ))
# eighth_dense = Dense(1, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(eighth_input)

# ninth_input = Input(shape=(2, ))
# ninth_dense = Dense(1, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(ninth_input)

# merge_one = concatenate([first_dense, second_dense, second_dense1, third_dense_1,third_dense, fouth_dense, 
#                          fifth_dense, sixth_dense, seventh_dense, eighth_dense, ninth_dense])

merge_one = concatenate([first_dense, second_dense, fouth_dense, 
                        sixth_dense, seventh_dense])


# 解码层
decoded = Dense(5, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(merge_one)
#decoded = Dropout(0.2)(decoded)
decoded = Dense(18, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3))(decoded)
#decoded = Dropout(0.2)(decoded)
decoded_output = Dense(20, activation='tanh')(decoded)

# model = Model(inputs=[first_input, second_input, second_input1, third_input_1,third_input, 
#                         fouth_input, fifth_input, sixth_input, seventh_input,
#                         eighth_input, ninth_input], outputs=decoded_output)
model = Model(inputs=[first_input, second_input, fouth_input, sixth_input, seventh_input], outputs=decoded_output)

# compile autoencoder
model.compile(optimizer='adam', loss='mse', metrics=[metrics.mse])
print(model.summary())
# plot_model(model,to_file='result/{}/model.png'.format(model_name),show_shapes=True)

if DO_TRAINING:
    weight_file_path = 'model/{}/{}'.format(model_name,model_name)+'-weights2.{epoch:02d}-{val_loss:.8f}.h5'
    architecture_file_path = 'model/{}/{}-architecture.json'.format(model_name,model_name)
    open(architecture_file_path, 'w').write(model.to_json())
    # training
    checkpoint = ModelCheckpoint(weight_file_path)
    # history = model.fit([x_train[:,0:1],x_train[:,1:3],x_train[:,3:4],x_train[:,4:11],x_train[:,11:15],x_train[:,15:17],x_train[:,17:19],x_train[:,19:30],x_train[:,30:32],x_train[:,32:34]], x_train_,
    #                 validation_data=([x_test[:,0:1],x_test[:,1:3],x_test[:,3:4],x_test[:,4:11],x_test[:,11:15],x_test[:,15:17],x_test[:,17:19],x_test[:,19:30],x_test[:,30:32],x_test[:,32:34]],x_test_), callbacks=[checkpoint],epochs=500, batch_size=10, shuffle=True)

    history = model.fit([x_train[:,0:1],x_train[:,1:3],x_train[:,11:15],x_train[:,17:19],x_train[:,19:30],x_train[:,32:34]], x_train_[:,[0,1,2,11,12,13,14,17,18,19,20,21,22,23,24,25,26,27,28,29,32,33]],
                    validation_data=([x_test[:,0:1],x_test[:,1:3],x_test[:,11:15],x_test[:,17:19],x_test[:,19:30],x_test[:,32:34]],x_test_[:,[0,1,2,11,12,13,14,17,18,19,20,21,22,23,24,25,26,27,28,29,32,33]]), callbacks=[checkpoint],epochs=500, batch_size=10, shuffle=True)


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    plt.savefig('result/{}/loss.png'.format(model_name))
else:
    weight_file_path = 'model/{}/{}.h5'.format(model_name,'autoencoder8-weights7.110-0.00019103')
    model.load_weights(weight_file_path)

# features = encoder.predict(satellite_np_data)
# data_features = pd.DataFrame(features,index=index)
# data_features.to_csv('result/{}/{}-feat3-370.csv'.format(model_name,model_name), encoding='utf-8')


# encoded_prd = model.predict([satellite_np_data[:,0:1],satellite_np_data[:,1:3],satellite_np_data[:,3:4],satellite_np_data[:,4:11],satellite_np_data[:,11:15],satellite_np_data[:,15:17],
# satellite_np_data[:,17:19],satellite_np_data[:,19:30],satellite_np_data[:,30:32],satellite_np_data[:,32:34]])
encoded_prd = model.predict([satellite_np_data[:,0:1],satellite_np_data[:,1:3],satellite_np_data[:,11:15],
satellite_np_data[:,17:19],satellite_np_data[:,19:30]])

data_target = pd.DataFrame(encoded_prd, index=index, columns=columns)
data_target.to_csv('result/{}/{}-prd7-110.csv'.format(model_name,model_name), encoding='utf-8')

# data_target = pd.DataFrame(encoded_prd, index=index, columns=columns)

# dataset_basic = data_target.as_matrix()
# # data_target.to_csv('result/{}/{}-ano7-249.csv'.format(model_name,model_name), encoding='utf-8')

# dist = np.linalg.norm(dataset_basic - satellite_np_data, axis=-1).reshape(-1,1)
# data_dist = pd.DataFrame(dist, index=index, columns=['norm'])
# data_dist.to_csv('result/{}/{}-ano7-249-2.csv'.format(model_name,model_name), encoding='utf-8')