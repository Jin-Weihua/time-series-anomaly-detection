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

##########################
# 只使用全连接Dense,正则自编码器--dropout
##########################

DO_TRAINING = True
model_name = 'autoencoder2'
dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
satellite_data = pd.read_csv(
    'data/data_rolling.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)
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
encoded = Dropout(0.2)(input_data)
encoded = Dense(18, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3))(encoded)
encoded = Dropout(0.2)(encoded)
encoder_output = Dense(9, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3))(encoded)
 
# 解码层
decoded = Dense(9, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3))(encoder_output)
decoded = Dropout(0.2)(decoded)
decoded = Dense(18, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3))(decoded)
decoded = Dropout(0.2)(decoded)
decoded_output = Dense(34, activation='selu')(decoded)
 
# 构建自编码模型
autoencoder = Model(inputs=input_data, outputs=decoded_output)
 
# 构建编码模型
encoder = Model(inputs=input_data, outputs=encoder_output)
 
# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mae', metrics=[metrics.mae])
print(autoencoder.summary())

if DO_TRAINING:
    weight_file_path = 'model/{}/{}'.format(model_name,model_name)+'-weights.{epoch:02d}-{val_loss:.8f}.h5'
    architecture_file_path = 'model/{}/{}-architecture.json'.format(model_name,model_name)
    open(architecture_file_path, 'w').write(autoencoder.to_json())
    # training
    checkpoint = ModelCheckpoint(weight_file_path)
    history = autoencoder.fit(x_train, x_train,validation_data=(x_test,x_test), callbacks=[checkpoint],epochs=200, batch_size=10, shuffle=True)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    plt.savefig('result/{}/loss.png'.format(model_name))
else:
    weight_file_path = 'model/{}/{}.h5'.format(model_name,'autoencoder2-weights.198-0.00662580')
    autoencoder.load_weights(weight_file_path)

# features = encoder.predict(satellite_np_data)
# data_features = pd.DataFrame(features,index=index)
# data_features.to_csv('result/{}/{}-feat.csv'.format(model_name,model_name), encoding='utf-8')

encoded_prd = autoencoder.predict(satellite_np_data)

data_target = pd.DataFrame(encoded_prd, index=index, columns=columns)
data_target.to_csv('result/{}/{}-prd.csv'.format(model_name,model_name), encoding='utf-8')