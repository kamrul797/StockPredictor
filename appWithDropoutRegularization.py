from sklearn.preprocessing  import MinMaxScaler;
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
from keras.models import Sequential;
from keras.layers import Dense;
from keras.layers import LSTM;
from tkinter import *
from keras.layers import Dropout;

# BEXIMCO

# Load Data and Convert Data Type
BEXIMCO_data=pd.read_csv(r'data/BEXIMCO.csv', index_col="DATE", parse_dates=True)
#BEXIMCO_data["OPEN"] = BEXIMCO_data["OPEN"].str.replace(',', '').astype(float);
#BEXIMCO_data["HIGH"] = BEXIMCO_data["HIGH"].str.replace(',', '').astype(float);
#BEXIMCO_data["LOW"] = BEXIMCO_data["LOW"].str.replace(',', '').astype(float);
#BEXIMCO_data["CLOSE"] = BEXIMCO_data["CLOSE"].str.replace(',', '').astype(float);
#BEXIMCO_data["VOLUME"] = BEXIMCO_data["VOLUME"].str.replace(',', '').astype(float);
BEXIMCO_data2 = BEXIMCO_data.reset_index()['CLOSE'];
#print(BEXIMCO_data.info())

# Convert/Scale the Data
scaler= MinMaxScaler(feature_range = (0,1))
BEXIMCO_data2 = scaler.fit_transform(np.array(BEXIMCO_data2).reshape(-1,1));

# Split Data into Test and Train sets
training_size=int(len(BEXIMCO_data2)*0.90);
test_size=len(BEXIMCO_data2)-training_size;
BEX_train_data, BEX_test_data = BEXIMCO_data2[0:training_size,:], BEXIMCO_data2[training_size:len(BEXIMCO_data2),:1];
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], [];
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a);
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY);    
time_step = 42;
BEX_X_train, BEX_y_train = create_dataset(BEX_train_data, time_step);
BEX_X_test, BEX_y_test = create_dataset(BEX_test_data, time_step);        
BEX_X_train = BEX_X_train.reshape(BEX_X_train.shape[0], BEX_X_train.shape[1], 1);
BEX_X_test = BEX_X_test.reshape(BEX_X_test.shape[0], BEX_X_test.shape[1], 1);

#---------------------------------------------------------------------------------

# BATBC

# Load Data and Convert Data Type
BATBC_data=pd.read_csv(r'data/BATBC.csv', index_col="DATE", parse_dates=True)
#BATBC_data["OPEN"] = BATBC_data["OPEN"].str.replace(',', '').astype(float);
#BATBC_data["HIGH"] = BATBC_data["HIGH"].str.replace(',', '').astype(float);
#BATBC_data["LOW"] = BATBC_data["LOW"].str.replace(',', '').astype(float);
BATBC_data["CLOSE"] = BATBC_data["CLOSE"].str.replace(',', '').astype(float);
#BATBC_data["VOLUME"] = BATBC_data["VOLUME"].str.replace(',', '').astype(float);
BATBC_data2 = BATBC_data.reset_index()['CLOSE'];
#print(BATBC_data.info())

# Convert/Scale the Data
scaler= MinMaxScaler(feature_range = (0,1))
BATBC_data2 = scaler.fit_transform(np.array(BATBC_data2).reshape(-1,1));

# Split Data into Test and Train sets
training_size=int(len(BATBC_data2)*0.70);
test_size=len(BATBC_data2)-training_size;
BAT_train_data, BAT_test_data = BATBC_data2[0:training_size,:], BATBC_data2[training_size:len(BATBC_data2),:1];

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], [];
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a);
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY);    

time_step = 42;
BAT_X_train, BAT_y_train = create_dataset(BAT_train_data, time_step);
BAT_X_test, BAT_y_test = create_dataset(BAT_test_data, time_step);        
BAT_X_train = BAT_X_train.reshape(BAT_X_train.shape[0], BAT_X_train.shape[1], 1);
BAT_X_test = BAT_X_test.reshape(BAT_X_test.shape[0], BAT_X_test.shape[1], 1);

# ---------------------------------------------------------------------------------------

# LANKABANGLA

# Load Data and Convert Data Type
LB_data=pd.read_csv(r'data/LANKABANGLA.csv', index_col="DATE", parse_dates=True)
#LB_data["OPEN"] = LB_data["OPEN"].str.replace(',', '').astype(float);
#LB_data["HIGH"] = LB_data["HIGH"].str.replace(',', '').astype(float);
#LB_data["LOW"] = LB_data["LOW"].str.replace(',', '').astype(float);
#LB_data["CLOSE"] = LB_data["CLOSE"].str.replace(',', '').astype(float);
#LB_data["VOLUME"] = LB_data["VOLUME"].str.replace(',', '').astype(float);
LB_data2 = LB_data.reset_index()['CLOSE'];
#print(BATBC_data.info())

# Convert/Scale the Data
scaler= MinMaxScaler(feature_range = (0,1))
LB_data2 = scaler.fit_transform(np.array(LB_data2).reshape(-1,1));

# Split Data into Test and Train sets
training_size=int(len(LB_data2)*0.70);
test_size=len(LB_data2)-training_size;
LB_train_data, LB_test_data = LB_data2[0:training_size,:], LB_data2[training_size:len(LB_data2),:1];

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], [];
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a);
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY);    

time_step = 42;
LB_X_train, LB_y_train = create_dataset(LB_train_data, time_step);
LB_X_test, LB_y_test = create_dataset(LB_test_data, time_step);        
LB_X_train = LB_X_train.reshape(LB_X_train.shape[0], LB_X_train.shape[1], 1);
LB_X_test = LB_X_test.reshape(LB_X_test.shape[0], LB_X_test.shape[1], 1);

# ---------------------------------------------------------------------------------------

# Build the LSTM Network
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

regressor.add(Dense(units=1));

regressor.compile(optimizer = 'Adam', loss= 'mean_squared_error');

l1=['BEXIMCO', 'BATBC', 'LANKABANGLA']

def LSTM():
    selection=Symptom1.get()
    if (selection=='BEXIMCO'):
        print('Total Data:', len(BEXIMCO_data2))
        print('TrainData:', len(BEX_train_data),',', 'TestData:', len(BEX_test_data));
        print('TimeStep:',time_step);
        
        # Train/Fit the Model
        regressor.fit(BEX_X_train, BEX_y_train, validation_data=(BEX_X_test,BEX_y_test), epochs=100, batch_size=64, verbose=1);
        train_predict=regressor.predict(BEX_X_train);
        test_predict=regressor.predict(BEX_X_test);
        train_predict=scaler.inverse_transform(train_predict);
        test_predict=scaler.inverse_transform(test_predict);
        x_input=BEX_test_data[8:].reshape(1,-1);
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist();
        
        # Make prediction for the next 10 days
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
                lst_output.extend(ythat.tolist())
                i=i+1; 
                
        # Visualization
        fig = plt.figure(figsize=(10,5))
        fig.suptitle('Stock Closing Price Prediction [BEXIMCO]', fontsize='12')
        data3=BEXIMCO_data2.tolist()
        data3.extend(lst_output) 
        plt.plot(data3[400:],color='green',alpha=0.9, label="Closing price of next 10 days [Prediction]")
        plt.plot(BEXIMCO_data2[400:],color='firebrick',alpha=0.9, label="Previous closing price [Train Data]")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.xlabel('Number of Days',fontsize='11.5')
        plt.ylabel('Stock Closing Price',fontsize='11.5')
        plt.show()
        fig.savefig('Output/BEXIMCO.jpg')

    elif (selection=='BATBC'):
        print('Total Data:', len(BATBC_data2))
        print('TrainData:', len(BAT_train_data),',', 'TestData:', len(BAT_test_data));
        print('TimeStep:',time_step);
        
        # Train/Fit the Model
        regressor.fit(BAT_X_train, BAT_y_train, validation_data=(BAT_X_test,BAT_y_test), epochs=100, batch_size=64, verbose=1);
        train_predict=regressor.predict(BAT_X_train);
        test_predict=regressor.predict(BAT_X_test);
        train_predict=scaler.inverse_transform(train_predict);
        test_predict=scaler.inverse_transform(test_predict);
        x_input=BAT_test_data[108:].reshape(1,-1);
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist();
        
        # Make prediction for the next 10 days
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
                lst_output.extend(ythat.tolist())
                i=i+1;        
                
        # Visualization        
        fig = plt.figure(figsize=(10,5))
        fig.suptitle('Stock Closing Price Prediction [BATBC]', fontsize='12')
        data3=BATBC_data2.tolist()
        data3.extend(lst_output) 
        plt.plot(data3[400:],color='green',alpha=0.9, label="Closing price of next 10 days [Prediction]")
        plt.plot(BATBC_data2[400:],color='firebrick',alpha=0.9, label="Previous closing price [Train Data]")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.xlabel('Number of Days',fontsize='11.5')
        plt.ylabel('Stock Closing Price',fontsize='11.5')
        plt.show()
        fig.savefig('Output/BATBC.jpg')

    elif (selection=='LANKABANGLA'):
        print('Total Data:', len(LB_data2))
        print('TrainData:', len(LB_train_data),',', 'TestData:', len(LB_test_data));
        print('TimeStep:',time_step);
        
        # Train/Fit the Model
        regressor.fit(LB_X_train, LB_y_train, validation_data=(LB_X_test,LB_y_test), epochs=100, batch_size=64, verbose=1);
        train_predict=regressor.predict(LB_X_train);
        test_predict=regressor.predict(LB_X_test);
        train_predict=scaler.inverse_transform(train_predict);
        test_predict=scaler.inverse_transform(test_predict);
        x_input=LB_test_data[108:].reshape(1,-1);
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist();
        
        # Make prediction for the next 10 days
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
                lst_output.extend(ythat.tolist())
                i=i+1;        
        
        # Visualization
        fig = plt.figure(figsize=(10,5))
        fig.suptitle('Stock Closing Price Prediction [LANKABANGLA]', fontsize='12')
        data3=LB_data2.tolist()
        data3.extend(lst_output) 
        plt.plot(data3[400:],color='green',alpha=0.9, label="Closing price of next 10 days [Prediction]")
        plt.plot(LB_data2[400:],color='firebrick',alpha=0.9, label="Previous closing price [Train Data]")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.xlabel('Number of Days',fontsize='11.5')
        plt.ylabel('Stock Closing Price',fontsize='11.5')
        plt.show()
        fig.savefig('Output/LANKABANGLA.jpg') 
        
# GUI Part
root = Tk()
root.wm_title("Stock Closing Price Predictor")
root.configure(background='white')
w2 = Label(root, text="Stock Closing Price Predictor", fg="Orange", bg="White")
w2.config(font=("Poppins",16,"bold"))
w2.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
Symptom1 = StringVar()
Symptom1.set("Choose Company")
S1Lb = Label(root, text="Company Name:", fg="Black", bg="White")
S1Lb.config(font=("Poppins",10,"bold"))
S1Lb.grid(row=7, column=0, padx=10, pady=10, sticky=W)
OPTIONS = sorted(l1)
S1 = OptionMenu(root, Symptom1,*OPTIONS)
S1.grid(row=7, column=1) 
lstm = Button(root, text="Predict", command=LSTM,bg="Orange",fg="white")
lstm.config(font=("poppins",10,"bold"))
lstm.grid(row=7, column=3, padx=10, pady=10)
root.mainloop()