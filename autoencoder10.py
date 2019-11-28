import numpy as np
np.random.seed(1337)  # for reproducibility
 
from keras.datasets import mnist
from keras.models import Sequential, Model #泛型模型
from keras.layers import Dense, Input, Dropout, LSTM, RepeatVector, TimeDistributed
from keras import metrics,regularizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.constraints import maxnorm
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model

##########################
# autoencoder10 LSTM预测
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

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

DO_TRAINING = False
model_name = 'autoencoder10'
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



satellite_np_data = satellite_data.as_matrix()
print(satellite_np_data.shape)

reframed_satellite_np_data = series_to_supervised(satellite_np_data, 5)
print(reframed_satellite_np_data.head(10))
time_step = 5
input_dim = satellite_data.shape[1]
values = reframed_satellite_np_data.values
train = values[:80000, :]
test = values[80000:96700:, :]

test_end = time_step*satellite_data.shape[1]

# split into input and outputs
train_X, train_y = train[:, :test_end], train[:, test_end:train.shape[1]]
test_X, test_y = test[:, :test_end], test[:, test_end:train.shape[1]]

# x_train, x_test = addNoise(train_X,test_X)
# y_train, y_test = addNoise(train_y,test_y)

train_X = train_X.reshape((train_X.shape[0], time_step, satellite_data.shape[1]))
test_X = test_X.reshape((test_X.shape[0], time_step, satellite_data.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

values_ = values[:96700, :test_end]
values_ = values_.reshape((values_.shape[0], time_step, satellite_data.shape[1]))

# satellite_np_data_ = np.reshape(
#             satellite_np_data[0:96700],
#             ((int)(satellite_np_data[0:96700].shape[0] / time_step),
#              time_step, satellite_np_data[0:96700].shape[1]))


# design network
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(input_dim))
model.compile(optimizer='adam', loss='mse', metrics=[metrics.mse])

index = satellite_data[5:96705].index
columns = satellite_data.columns

print(model.summary())
plot_model(model,to_file='result/{}/model.svg'.format(model_name),show_shapes=True)

if DO_TRAINING:
    weight_file_path = 'model/{}/{}'.format(model_name,model_name)+'-weights.{epoch:02d}-{val_loss:.8f}.h5'
    architecture_file_path = 'model/{}/{}-architecture.json'.format(model_name,model_name)
    open(architecture_file_path, 'w').write(model.to_json())
    # training
    checkpoint = ModelCheckpoint(weight_file_path)

    history = model.fit(train_X, train_y,
                    validation_data=(test_X, test_y), callbacks=[checkpoint],epochs=2, batch_size=10, shuffle=False)


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    plt.savefig('result/{}/loss.png'.format(model_name))
else:
    weight_file_path = 'model/{}/{}.h5'.format(model_name,'autoencoder10-weights2.99-0.00404273')
    model.load_weights(weight_file_path)

# features = encoder.predict(satellite_np_data)
# data_features = pd.DataFrame(features,index=index)
# data_features.to_csv('result/{}/{}-feat3-370.csv'.format(model_name,model_name), encoding='utf-8')

encoded_prd = model.predict(values_,batch_size=10)
encoded_prd_ = np.reshape(encoded_prd,(96700,34))
data_target = pd.DataFrame(encoded_prd_, index=index, columns=columns)
data_target.to_csv('result/{}/{}-prd2-99.csv'.format(model_name,model_name), encoding='utf-8')

# dataset_basic = data_target.as_matrix()
# # data_target.to_csv('result/{}/{}-ano7-249.csv'.format(model_name,model_name), encoding='utf-8')

# dist = np.linalg.norm(dataset_basic - satellite_np_data, axis=-1).reshape(-1,1)
# data_dist = pd.DataFrame(dist, index=index, columns=['norm'])
# data_dist.to_csv('result/{}/{}-ano7-249-2.csv'.format(model_name,model_name), encoding='utf-8')