import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dropout, Dense, LSTM, GRU, Bidirectional
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import math
data = pd.read_csv('dataset/London_Array.csv',index_col='time')

data = data.fillna(0)
print(data)
# print(df.head())
data.index = pd.to_datetime(data.index)
#特征热度图
# plt.subplots(figsize=(16, 16))
# sns.heatmap(df.corr(), annot=True, square=True)
# plt.show()
predict_len = 1

start_test = 28000

train, test = data.iloc[:start_test], data.iloc[start_test:]

# print(data)
# print(train)
SCALER = MinMaxScaler(feature_range=(-1,1))
# print(test.to_numpy())

scaler = SCALER.fit(train.to_numpy())

train_scaled = scaler.transform(train.to_numpy())

test_scaled = scaler.transform(test.to_numpy())

train_len = len(train_scaled)
test_len = len(test_scaled)
train_x , train_y = train_scaled[:train_len - predict_len,0:2], train_scaled[predict_len:,-1]
test_x, test_y = test_scaled[: test_len - predict_len,0:2], test_scaled[predict_len :,-1]



units = 64

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units, activation='relu', input_shape=(train_x.shape[1],)),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error', metrics=['accuracy'])

cp_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(train_x, train_y, batch_size=48, epochs=100, validation_data=(test_x, test_y), validation_freq=1,
          callbacks=cp_callback)

predicted = model.predict(test_x)

predicted = np.array(predicted).reshape(-1)


def prior_inverse(features, targets):
    '''
    Append prediction value to test dataset and return a test shape format.
    '''
    dataset = []

    for i in range(features.shape[0]):
        last_row, target = features[i], targets[i]
        appended = np.append(target,last_row)
        dataset.append(appended)

    return np.array(dataset)

pred = prior_inverse(test_x, predicted)
# print(pred)
real = prior_inverse(test_x, test_y)
inv_pred = scaler.inverse_transform(pred)
inv_real = scaler.inverse_transform(real)
plt.figure(figsize=(10,5))
plt.plot(inv_pred[:200,0], label='predict')
plt.plot(inv_real[:200,0], label='real')
plt.legend()
plt.title('Predicted Power vs Actual Power with MLP an Model.')
plt.tight_layout()
plt.show()



x_plot = test.iloc[predict_len:,:].index

result = pd.DataFrame({'Date':x_plot, 'Prediction':inv_pred[:,0], 'True':inv_real[:,0]})
# result.to_csv("result1/MLPlow6_1h.csv")
result.set_index('Date', inplace=True)

result2 = result['2020-10-01 00:00:00':'2020-10-07 00:00:00']
result2.plot(rot='60',figsize=(10,5))
plt.title('Predicted Power vs Actual Power with MLP an Model.')
plt.ylabel('Power(KWh)')
plt.tight_layout()
plt.show()

rmse = np.sqrt(mean_squared_error(inv_real[:,0], inv_pred[:,0]))
mae = mean_absolute_error(inv_real[:,0], inv_pred[:,0])
r2 = r2_score(inv_real[:,0], inv_pred[:,0])
print('RMSE: {}\nMAE: {}\nR2: {}'.format(round(rmse,2),round(mae,2), round(r2,2)))

