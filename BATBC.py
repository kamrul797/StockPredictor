from sklearn.preprocessing  import MinMaxScaler;
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
from keras.models import Sequential;
from keras.layers import Dense;
from keras.layers import LSTM;
import tkinter
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler


data=pd.read_csv(r'BATBC.csv', index_col="DATE", parse_dates=True)
#data["OPEN"] = data["OPEN"].str.replace(',', '').astype(float);
#data["HIGH"] = data["HIGH"].str.replace(',', '').astype(float);
#data["LOW"] = data["LOW"].str.replace(',', '').astype(float);
data["CLOSE"] = data["CLOSE"].str.replace(',', '').astype(float);
#data["VOLUME"] = data["VOLUME"].str.replace(',', '').astype(float);
data2 = data.reset_index()['CLOSE'];
print(data.info())
#print(data2)

scaler= MinMaxScaler(feature_range = (0,1))
data2 = scaler.fit_transform(np.array(data2).reshape(-1,1));
#print(data2);

training_size=int(len(data2)*0.70);
test_size=len(data2)-training_size;
train_data, test_data = data2[0:training_size,:], data2[training_size:len(data2),:1];
print('Total Data:', len(data2))
print('TrainData:', len(train_data),',', 'TestData:', len(test_data));
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], [];
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a);
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY);    
time_step = 42;
print('TimeStep:',time_step);
X_train, y_train = create_dataset(train_data, time_step);
X_test, y_test = create_dataset(test_data, time_step);        
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1);
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1);

regressor= Sequential();
#1st LSTM Layer
regressor.add(LSTM(50, return_sequences=True, input_shape = (42,1)));

#2nd LSTM Layer
regressor.add(LSTM(50, return_sequences=True));

#3rd LSTM Layer
regressor.add(LSTM(50))

regressor.add(Dense(1))
regressor.compile(loss='mean_squared_error', optimizer='adam');

#print(regressor.summary())
regressor.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=64, verbose=1);
train_predict=regressor.predict(X_train);
test_predict=regressor.predict(X_test);
train_predict=scaler.inverse_transform(train_predict);
test_predict=scaler.inverse_transform(test_predict);

x_input=test_data[108:].reshape(1,-1);
#print(x_input.shape)
temp_input=list(x_input)
temp_input=temp_input[0].tolist();

#10 days prediction
lst_output=[]
n_steps=42;
i=0
while(i<10):
    if(len(temp_input)>42):
        x_input=np.array(temp_input[1:])
        print("{} day input{}".format(i,x_input));
        x_input=x_input.reshape(1,-1);
        x_input=x_input.reshape((1, n_steps,1))
        ythat=regressor.predict(x_input, verbose=0)
        print("{} day output{}".format(i,ythat));
        temp_input.extend(ythat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(ythat.tolist())
        i=i+1;   
    else:
        x_input=x_input.reshape((1, n_steps,1))
        ythat=regressor.predict(x_input, verbose=0)
        print(ythat[0])
        temp_input.extend(ythat[0].tolist());
        #print(len(temp_input))
        lst_output.extend(ythat.tolist())
        i=i+1;        
#print(lst_output)
fig = plt.figure(figsize=(10,5))
fig.suptitle('Stock Closing Price Prediction [BATBC]', fontsize='12')
data3=data2.tolist()
data3.extend(lst_output) 
plt.plot(data3[400:],color='green',alpha=0.9, label="Closing price of next 10 days [Prediction]")
plt.plot(data2[400:],color='firebrick',alpha=0.9, label="Previous closing price [Train Data]")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('Number of Days',fontsize='11.5')
plt.ylabel('Stock Closing Price',fontsize='11.5')
plt.show()
