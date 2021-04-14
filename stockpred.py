import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense ,LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
df = web.DataReader('BAJFINANCE.NS',data_source='yahoo',start = '2019-11-01',end = '2020-12-27')
print(df)
print(df.shape)
plt.figure(figsize=(16,8))
plt.title('close price history')
plt.plot(df['Close'])
plt.xlabel('Date')
plt.ylabel('RUPEES')
plt.show()
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)
print(training_data_len)
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)
train_data = scaled_data[0:training_data_len , :]
x_train = []
y_train = []
for i in range(10,len(train_data)):
  x_train.append(train_data[i-10:i, 0])
  y_train.append(train_data[i,0])
  if i<=11:
    print(x_train)
    print(y_train)
    print()
 
   

#print(x_train)
#print(y_train)
x_train ,y_train = np.array(x_train),np.array(y_train)
print(x_train.shape)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
print(x_train.shape)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=1,epochs=1)
test_data = scaled_data[training_data_len - 10: , :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(10,len(test_data)):
  x_test.append(test_data[i-10:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()
print(valid)
