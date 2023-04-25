# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras.layers import Dropout, Dense, LSTM, BatchNormalization
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import math
from scipy.stats import pearsonr
#from keras import regularizers
from tensorflow.python.keras.callbacks import EarlyStopping

dataset = pd.read_csv("dataset/London_Array.csv", header=0)


time_interval = 24
n_feature = 1
time_step = 74
cell = 24
learning_rate = 0.001
batch_size = 64
epochs = 100

data = dataset.iloc[:len(dataset), 5:6].values


x = []
y = []
for i in range(time_step, len(data)-time_interval+1):
    x.append(data[i-time_step:i])
    y.append(data[i+time_interval-1])
x = np.array(x)
y = np.array(y)


train_volume = int(len(x) * 0.64)
val_volume = int(len(x) * 0.16)
test_volume = len(x) - train_volume - val_volume
test_num = test_volume


train_x = x[:train_volume, :]
val_x = x[train_volume:train_volume+val_volume, :]
test_x = x[-test_volume:, :]
train_y = y[:train_volume]
val_y = y[train_volume:train_volume+val_volume]
test_y = y[-test_volume:]

train_x, train_y = np.array(train_x), np.array(train_y)
train_x = np.reshape(train_x, (train_x.shape[0], time_step, n_feature))
val_x, val_y = np.array(val_x), np.array(val_y)
val_x = np.reshape(val_x, (val_x.shape[0], time_step, n_feature))
test_x, test_y = np.array(test_x), np.array(test_y)
test_x = np.reshape(test_x, (test_x.shape[0], time_step, n_feature))
test_y = np.array(test_y)


model = tf.keras.Sequential([
     LSTM(cell, activation='tanh', input_shape=(time_step, n_feature), return_sequences=False),
     Dense(cell),
     Dense(1)
    ])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mean_absolute_error',
                  metrics=['mae'])
cp_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y),
                        validation_freq=1, callbacks=cp_callback)
model.summary()


predicted_data = model.predict(test_x)

predicted_wave = []
for i in predicted_data:
    for j in i:
        predicted_wave.append(j)

predict_wind_power = []
for i in range(len(predicted_wave)):
    predict_wind_power.append(predicted_wave[i])





mse = mean_squared_error(predict_wind_power, test_y)
rmse = math.sqrt(mean_squared_error(predict_wind_power, test_y))
mae = mean_absolute_error(predict_wind_power, test_y)
mape = np.mean(np.abs((test_y - predict_wind_power) / test_y))
R, p = pearsonr(test_y, predict_wind_power)
R2 = r2_score(test_y, predict_wind_power)

print('mse: %.4f' % mse)
print('rmse: %.4f' % rmse)
print('mae: %.4f' % mae)
print('mape: %.4f' % mape)
# print("R: %.4f" % R)
print("R2: %.4f" % R2)

# 预测效果图
fig = plt.figure(figsize=(20, 5))
plt.plot(test_y[0:800], color='red', label='real')
plt.plot(predict_wind_power[0:800], color='blue', label='LSTM')
plt.title('2014.11.30 2:50-2015.1.2 9:50')
plt.xlabel('Time Serise')
plt.ylabel('Wind Power')
plt.legend()
plt.show()
pred_df = pd.DataFrame({'Prediction': predict_wind_power.flatten(), 'True': test_y.flatten()})



