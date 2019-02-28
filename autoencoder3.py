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
# autoencoder2 中提取的特征有问题，测试使用分布训练自编码器，
# 先训练一个特征，训练成功后，加入另外一个特征，直至训练完成
##########################

DO_TRAINING = False
model_name = 'autoencoder3'
dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
satellite_data = pd.read_csv(
    'data/data_rolling.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)
# ['VNZ4A组蓄电池BEA信号','VNZ5B组蓄电池BEA信号','INZ6_-Y太阳电池阵电流','INZ7_+Y太阳电池阵电流']
for column in satellite_data.columns:
   if column not in ['VNZ4A组蓄电池BEA信号','VNZ5B组蓄电池BEA信号','INZ6_-Y太阳电池阵电流','INZ7_+Y太阳电池阵电流','INZ14_ABCR1输入电流','INZ15_ABCR2输入电流']:
       satellite_data[column] = 0.5
satellite_np_data = satellite_data.as_matrix()
print(satellite_np_data.shape)
index = satellite_data.index
columns = satellite_data.columns

x_train = satellite_np_data[0:80000]
x_test = satellite_np_data[80000:96700]
print(x_train.shape)
print(x_test.shape)
 
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

#weight_file_path = 'model/{}/{}.h5'.format(model_name,'autoencoder3-weights1.148-0.00020910')
#autoencoder.load_weights(weight_file_path)

#weight_file_path = 'model/{}/{}.h5'.format(model_name,'autoencoder3-weights2.118-0.00034362')
#autoencoder.load_weights(weight_file_path)

#weight_file_path = 'model/{}/{}.h5'.format(model_name,'autoencoder3-weights3.112-0.00039538')
#autoencoder.load_weights(weight_file_path)

if DO_TRAINING:
    weight_file_path = 'model/{}/{}'.format(model_name,model_name)+'-weights3.{epoch:02d}-{val_loss:.8f}.h5'
    architecture_file_path = 'model/{}/{}-architecture.json'.format(model_name,model_name)
    open(architecture_file_path, 'w').write(autoencoder.to_json())
    # training
    checkpoint = ModelCheckpoint(weight_file_path)
    history = autoencoder.fit(x_train, x_train, validation_data=(x_test,x_test), callbacks=[checkpoint],epochs=200, batch_size=10, shuffle=True)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    plt.savefig('result/{}/loss.png'.format(model_name))
else:
    weight_file_path = 'model/{}/{}.h5'.format(model_name,'autoencoder3-weights3.112-0.00039538)
    autoencoder.load_weights(weight_file_path)

# features = encoder.predict(satellite_np_data)
# data_features = pd.DataFrame(features,index=index)
# data_features.to_csv('result/{}/{}-feat.csv'.format(model_name,model_name), encoding='utf-8')

encoded_prd = autoencoder.predict(satellite_np_data)

data_target = pd.DataFrame(encoded_prd, index=index, columns=columns)
data_target.to_csv('result/{}/{}-prd3.csv'.format(model_name,model_name), encoding='utf-8')