from sklearn.preprocessing  import MinMaxScaler;
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
import datetime;
from keras.models import Sequential;
from keras.layers import Dense;
from keras.layers import LSTM;
from keras.layers import Dropout;

data=pd.read_csv(r'SQPHARMA.csv', index_col="DATE", parse_dates=True)
head = data.head();
#print(head);
#tail = data.tail();
#print(tail);
#data.info();

#data preprocessing

data["OPEN"] = data["OPEN"].str.replace(',', '').astype(float);
data["HIGH"] = data["HIGH"].str.replace(',', '').astype(float);
data["LOW"] = data["LOW"].str.replace(',', '').astype(float);
data["CLOSE"] = data["CLOSE"].str.replace(',', '').astype(float);
data["VOLUME"] = data["VOLUME"].str.replace(',', '').astype(float);
#data.info();
#data['CLOSE'].plot(figsize=(16,6))
data.rolling(window=15).mean()['CLOSE'].plot()


#data cleaning
#st = data.isna().any;
#print(st);

#data scaling
train_set=data['CLOSE']
train_set=pd.DataFrame(train_set);
scaler= MinMaxScaler(feature_range = (0,1))
scaled_train_set = scaler.fit_transform(train_set)

#train set for 15 timesteps
X_train=[]
y_train=[]
for i in range(15,397):
    X_train.append(scaled_train_set[i-15:i,0])
    y_train.append(scaled_train_set[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
#print(X_train)
regressor= Sequential();

#1st LSTM Layer and dropout regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape = (X_train.shape[1], 1)));
regressor.add(Dropout(0.2));

#2nd LSTM Layer and dropout regularization
regressor.add(LSTM(units=50, return_sequences=True));
regressor.add(Dropout(0.2));

#3rd LSTM Layer and dropout regularization
regressor.add(LSTM(units=50, return_sequences=True));
regressor.add(Dropout(0.2));

#4th LSTM Layer and dropout regularization
regressor.add(LSTM(units=50));
regressor.add(Dropout(0.2));

#Output Layer
regressor.add(Dense(units=1));

regressor.compile(optimizer = 'Adam', loss= 'mean_squared_error');

regressor.fit( X_train, y_train, epochs=100, batch_size=32);

dataset_test= pd.read_csv('data.csv', index_col='DATE', parse_dates= True);
real_stock_price=dataset_test.iloc[:, 1:2].values;
print(dataset_test.info());
dataset_test["CLOSE"] = dataset_test["CLOSE"].str.replace(',' , '').astype(float);
test_set=dataset_test['CLOSE'];
test_set=pd.DataFrame(test_set);
test_set.info();

#getting the prediction
dataset_total = pd.concat((data['CLOSE'], dataset_test['CLOSE']), axis=0);
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values;
inputs = inputs.reshape(-1,1);
inputs = scaler.transform(inputs);
X_test = [];
for i in range (15,397):
    X_test.append(inputs[i-15:i, 0])
X_test =np.array(X_test);
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1));
predicted_stock_price = regressor.predict(X_test);
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
predicted_stock_price = pd.DataFrame(predicted_stock_price)
predicted_stock_price.info();

#Visualization
plt.plot(real_stock_price, color ='red', label ='Real Stock Price');
plt.plot(predicted_stock_price, color= 'blue', label = 'Predicted Stock Price');
plt.title('Stock Price Prediction');
plt.xlabel('Time');
plt.ylabel('Stock Price');
plt.legend();
plt.show();





