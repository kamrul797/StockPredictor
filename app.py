from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tkinter import *

# Create Dataset [Read Data from excel sheet]
BATBC = pd.read_excel('E:\IIT\Final Project\Code\App\Data\Latest_Data.xlsx',sheet_name=0, header=0, index_col="DATE")
BEXIMCO = pd.read_excel('E:\IIT\Final Project\Code\App\Data\Latest_Data.xlsx',sheet_name=1, header=0, index_col="DATE")
LANKABANGLA = pd.read_excel('E:\IIT\Final Project\Code\App\Data\Latest_Data.xlsx',sheet_name=2, header=0, index_col="DATE")
SAIFPOWER = pd.read_excel('E:\IIT\Final Project\Code\App\Data\Latest_Data.xlsx',sheet_name=4, header=0, index_col="DATE")
BXPHARMA = pd.read_excel('E:\IIT\Final Project\Code\App\Data\Latest_Data.xlsx',sheet_name=5, header=0, index_col="DATE")

#BEXIMCO
#Select Feature from Dataset
BEXIMCO_data1 = BEXIMCO["CLOSEP*"].iloc[::-1]

# Convert / Scale the Data
scaler= MinMaxScaler(feature_range = (0,1))
BEXIMCO_data2 = scaler.fit_transform(np.array(BEXIMCO_data1).reshape(-1,1))

# Split Data into Test and Train sets
training_size=int(len(BEXIMCO_data2)*0.90)
test_size=len(BEXIMCO_data2)-training_size
BEX_train_data, BEX_test_data = BEXIMCO_data2[0:training_size,:], BEXIMCO_data2[training_size:len(BEXIMCO_data2),:1]

# Create Test and Trainset method
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)  

#Define Time Step
time_step = 10

# Create and reshape Test and Trainset
BEX_X_train, BEX_y_train = create_dataset(BEX_train_data, time_step)
BEX_X_test, BEX_y_test = create_dataset(BEX_test_data, time_step)        
BEX_X_train = BEX_X_train.reshape(BEX_X_train.shape[0], BEX_X_train.shape[1], 1)
BEX_X_test = BEX_X_test.reshape(BEX_X_test.shape[0], BEX_X_test.shape[1], 1)

#---------------------------------------------------------------------------------

# BATBC
#Select Feature from Dataset
BATBC_data1 = BATBC["CLOSEP*"].iloc[::-1]

# Convert / Scale the Data
scaler= MinMaxScaler(feature_range = (0,1))
BATBC_data2 = scaler.fit_transform(np.array(BATBC_data1).reshape(-1,1))

# Split Data into Test and Train sets
training_size=int(len(BATBC_data2)*0.90)
test_size=len(BATBC_data2)-training_size
BAT_train_data, BAT_test_data = BATBC_data2[0:training_size,:], BATBC_data2[training_size:len(BATBC_data2),:1]

# Create Test and Trainset method
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)  

#Define Time Step
time_step = 10

# Create and reshape Test and Trainset
BAT_X_train, BAT_y_train = create_dataset(BAT_train_data, time_step)
BAT_X_test, BAT_y_test = create_dataset(BAT_test_data, time_step)        
BAT_X_train = BAT_X_train.reshape(BAT_X_train.shape[0], BAT_X_train.shape[1], 1)
BAT_X_test = BAT_X_test.reshape(BAT_X_test.shape[0], BAT_X_test.shape[1], 1)

# ---------------------------------------------------------------------------------------

# LANKABANGLA
#Select Feature from Dataset
LANKABANGLA_data1 = LANKABANGLA["CLOSEP*"].iloc[::-1]

# Convert / Scale the Data
scaler= MinMaxScaler(feature_range = (0,1))
LB_data2 = scaler.fit_transform(np.array(LANKABANGLA_data1).reshape(-1,1))

# Split Data into Test and Train sets
training_size=int(len(LB_data2)*0.90)
test_size=len(LB_data2)-training_size
LB_train_data, LB_test_data = LB_data2[0:training_size,:], LB_data2[training_size:len(LB_data2),:1]

# Create Test and Trainset method
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)    

#Define Time Step
time_step = 10

# Create and reshape Test and Trainset
LB_X_train, LB_y_train = create_dataset(LB_train_data, time_step)
LB_X_test, LB_y_test = create_dataset(LB_test_data, time_step)        
LB_X_train = LB_X_train.reshape(LB_X_train.shape[0], LB_X_train.shape[1], 1)
LB_X_test = LB_X_test.reshape(LB_X_test.shape[0], LB_X_test.shape[1], 1)

# ---------------------------------------------------------------------------------------

# SAIFPOWER
#Select Feature from Dataset
SAIFPOWER_data1 = SAIFPOWER["CLOSEP*"].iloc[::-1]

# Convert / Scale the Data
scaler= MinMaxScaler(feature_range = (0,1))
SP_data2 = scaler.fit_transform(np.array(SAIFPOWER_data1).reshape(-1,1))

# Split Data into Test and Train sets
training_size=int(len(SP_data2)*0.90)
test_size=len(SP_data2)-training_size
SP_train_data, SP_test_data = SP_data2[0:training_size,:], SP_data2[training_size:len(SP_data2),:1]

# Create Test and Trainset method
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)    

#Define Time Step
time_step = 10

# Create and reshape Test and Trainset
SP_X_train, SP_y_train = create_dataset(SP_train_data, time_step)
SP_X_test, SP_y_test = create_dataset(SP_test_data, time_step)        
SP_X_train = SP_X_train.reshape(SP_X_train.shape[0], SP_X_train.shape[1], 1)
SP_X_test = SP_X_test.reshape(SP_X_test.shape[0], SP_X_test.shape[1], 1)

# ---------------------------------------------------------------------------------------

# BXPHARMA
#Select Feature from Dataset
BXPHARMA_data1 = BXPHARMA["CLOSEP*"].iloc[::-1]

# Convert / Scale the Data
scaler= MinMaxScaler(feature_range = (0,1))
BP_data2 = scaler.fit_transform(np.array(BXPHARMA_data1).reshape(-1,1))

# Split Data into Test and Train sets
training_size=int(len(BP_data2)*0.90)
test_size=len(BP_data2)-training_size
BP_train_data, BP_test_data = BP_data2[0:training_size,:], BP_data2[training_size:len(BP_data2),:1]

# Create Test and Trainset method
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)    

#Define Time Step
time_step = 10

# Create and reshape Test and Trainset
BP_X_train, BP_y_train = create_dataset(BP_train_data, time_step)
BP_X_test, BP_y_test = create_dataset(BP_test_data, time_step)        
BP_X_train = BP_X_train.reshape(BP_X_train.shape[0], BP_X_train.shape[1], 1)
BP_X_test = BP_X_test.reshape(BP_X_test.shape[0], BP_X_test.shape[1], 1)

# ---------------------------------------------------------------------------------------

#finding the right epoch [Check if the system is overfit/underfit/perfectly fit]
#model = Sequential()
#model.add(LSTM(10, return_sequences=True, input_shape = (time_step,1)))
#model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(10))
#model.add(Dense(1))
#model.compile(loss='mse', optimizer='adam')
#history = model.fit(BEX_X_train, BEX_y_train, epochs=11, validation_data=(BEX_X_test,BEX_y_test), shuffle=False)
#plot train and validation loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model train vs validation loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper right')
#plt.show()

# Build the LSTM Network
regressor= Sequential()

# 1st LSTM Layer
regressor.add(LSTM(50, return_sequences=True, input_shape = (time_step,1)))

# 2nd LSTM Layer
regressor.add(LSTM(50, return_sequences=True))

# 3rd LSTM Layer
regressor.add(LSTM(50))

# Output Layer
regressor.add(Dense(1))

#Compile the model
regressor.compile(loss='mean_squared_error', optimizer='adam')

# Options for Dropdown
l1=['BEXIMCO', 'BATBC', 'LANKABANGLA', 'SAIFPOWER', 'BXPHARMA']
pred_day = ['5','10']

# ---------------------------------------------------------------------------------------

def LSTM():
    selection=Company.get()
    if (selection=='BEXIMCO'):
        print('Total Data:', len(BEXIMCO_data2))
        print('TrainData:', len(BEX_train_data),',', 'TestData:', len(BEX_test_data))
        print('TimeStep:',time_step)
        
        # Train/Fit and Test the Model
        regressor.fit(BEX_X_train, BEX_y_train, validation_data=(BEX_X_test,BEX_y_test), epochs=11, batch_size=64, verbose=1)
        train_predict=regressor.predict(BEX_X_train)
        test_predict=regressor.predict(BEX_X_test)
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)
        step=len(BEX_test_data)-time_step
        x_input=BEX_test_data[step:].reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        
        # Make prediction for the next 10 days
        lst_output=[]
        n_steps=time_step
        i=0
        days=int(Days.get())
        while(i<days):
            if(len(temp_input)>n_steps):
                x_input=np.array(temp_input[1:])
                #print("{} day input{}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print("{} day output{}".format(i,ythat))
                temp_input.extend(ythat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(ythat.tolist())
                i=i+1   
            else:
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print(ythat[0])
                temp_input.extend(ythat[0].tolist())
                lst_output.extend(ythat.tolist())
                i=i+1  
        
        # Visualization
        fig = plt.figure(figsize=(10,5))
        fig.suptitle('Stock Closing Price Prediction [BEXIMCO]', fontsize='12')
        data3=BEXIMCO_data2.tolist()
        data3.extend(lst_output) 
        plt.plot(data3[:],color='green',alpha=0.9, label="Closing price of next 10 days [Prediction]")
        plt.plot(BEXIMCO_data2[:],color='firebrick',alpha=0.9, label="Previous closing price [Train Data]")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.xlabel('Number of Days',fontsize='11.5')
        plt.ylabel('Stock Closing Price',fontsize='11.5')
        plt.show()
        if (days==5):
            fig.savefig('Output/BEXIMCO-5 days.jpg')
        else:
            fig.savefig('Output/BEXIMCO-10 days.jpg')

    elif (selection=='BATBC'):
        print('Total Data:', len(BATBC_data2))
        print('TrainData:', len(BAT_train_data),',', 'TestData:', len(BAT_test_data))
        print('TimeStep:',time_step)
        
        # Train/Fit and Test the Model
        regressor.fit(BAT_X_train, BAT_y_train, validation_data=(BAT_X_test,BAT_y_test), epochs=90, batch_size=64, verbose=1)
        train_predict=regressor.predict(BAT_X_train)
        test_predict=regressor.predict(BAT_X_test)
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)
        step=len(BAT_test_data)-time_step
        x_input=BAT_test_data[step:].reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        
        # Make prediction for the next 10 days
        lst_output=[]
        n_steps=time_step
        i=0
        days=int(Days.get())
        while(i<days):
            if(len(temp_input)>n_steps):
                x_input=np.array(temp_input[1:])
                #print("{} day input{}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print("{} day output{}".format(i,ythat))
                temp_input.extend(ythat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(ythat.tolist())
                i=i+1   
            else:
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print(ythat[0])
                temp_input.extend(ythat[0].tolist())
                lst_output.extend(ythat.tolist())
                i=i+1        
                
        # Visualization        
        fig = plt.figure(figsize=(10,5))
        fig.suptitle('Stock Closing Price Prediction [BATBC]', fontsize='12')
        data3=BATBC_data2.tolist()
        data3.extend(lst_output) 
        plt.plot(data3[:],color='green',alpha=0.9, label="Closing price of next 10 days [Prediction]")
        plt.plot(BATBC_data2[:],color='firebrick',alpha=0.9, label="Previous closing price [Train Data]")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.xlabel('Number of Days',fontsize='11.5')
        plt.ylabel('Stock Closing Price',fontsize='11.5')
        plt.show()
        if (days==5):
            fig.savefig('Output/BATBC-5 days.jpg')
        else:
            fig.savefig('Output/BATBC-10 days.jpg')

    elif (selection=='LANKABANGLA'):
        print('Total Data:', len(LB_data2))
        print('TrainData:', len(LB_train_data),',', 'TestData:', len(LB_test_data))
        print('TimeStep:',time_step)
        
        # Train/Fit and Test the Model
        regressor.fit(LB_X_train, LB_y_train, validation_data=(LB_X_test,LB_y_test), epochs=11, batch_size=64, verbose=1)
        train_predict=regressor.predict(LB_X_train)
        test_predict=regressor.predict(LB_X_test)
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)
        step=len(LB_test_data)-time_step
        x_input=LB_test_data[step:].reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        
        # Make prediction for the next 10 days
        lst_output=[]
        n_steps=time_step
        i=0
        days=int(Days.get())
        while(i<days):
            if(len(temp_input)>n_steps):
                x_input=np.array(temp_input[1:])
                #print("{} day input{}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print("{} day output{}".format(i,ythat))
                temp_input.extend(ythat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(ythat.tolist())
                i=i+1   
            else:
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print(ythat[0])
                temp_input.extend(ythat[0].tolist())
                lst_output.extend(ythat.tolist())
                i=i+1        
        
        # Visualization
        fig = plt.figure(figsize=(10,5))
        fig.suptitle('Stock Closing Price Prediction [LANKABANGLA]', fontsize='12')
        data3=LB_data2.tolist()
        data3.extend(lst_output) 
        plt.plot(data3[:],color='green',alpha=0.9, label="Closing price of next 10 days [Prediction]")
        plt.plot(LB_data2[:],color='firebrick',alpha=0.9, label="Previous closing price [Train Data]")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.xlabel('Number of Days',fontsize='11.5')
        plt.ylabel('Stock Closing Price',fontsize='11.5')
        plt.show()
        if (days==5):
            fig.savefig('Output/LANKABANGLA-5 days.jpg')
        else:
            fig.savefig('Output/LANKABANGLA-10 days.jpg')
            
    elif (selection=='SAIFPOWER'):
        print('Total Data:', len(SP_data2))
        print('TrainData:', len(SP_train_data),',', 'TestData:', len(SP_test_data))
        print('TimeStep:',time_step)
        
        # Train/Fit and Test the Model
        regressor.fit(SP_X_train, SP_y_train, validation_data=(SP_X_test,SP_y_test), epochs=11, batch_size=64, verbose=1)
        train_predict=regressor.predict(SP_X_train)
        test_predict=regressor.predict(SP_X_test)
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)
        step=len(SP_test_data)-time_step
        x_input=SP_test_data[step:].reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        
        # Make prediction for the next 10 days
        lst_output=[]
        n_steps=time_step
        i=0
        days=int(Days.get())
        while(i<days):
            if(len(temp_input)>n_steps):
                x_input=np.array(temp_input[1:])
                #print("{} day input{}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print("{} day output{}".format(i,ythat))
                temp_input.extend(ythat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(ythat.tolist())
                i=i+1   
            else:
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print(ythat[0])
                temp_input.extend(ythat[0].tolist())
                lst_output.extend(ythat.tolist())
                i=i+1        
        
        # Visualization
        fig = plt.figure(figsize=(10,5))
        fig.suptitle('Stock Closing Price Prediction [SAIFPOWER]', fontsize='12')
        data3=SP_data2.tolist()
        data3.extend(lst_output) 
        plt.plot(data3[:],color='green',alpha=0.9, label="Closing price of next 10 days [Prediction]")
        plt.plot(SP_data2[:],color='firebrick',alpha=0.9, label="Previous closing price [Train Data]")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.xlabel('Number of Days',fontsize='11.5')
        plt.ylabel('Stock Closing Price',fontsize='11.5')
        plt.show()
        if (days==5):
            fig.savefig('Output/SAIFPOWER-5 days.jpg')
        else:
            fig.savefig('Output/SAIFPOWER-10 days.jpg')    
    
    elif (selection=='BXPHARMA'):
        print('Total Data:', len(BP_data2))
        print('TrainData:', len(BP_train_data),',', 'TestData:', len(BP_test_data))
        print('TimeStep:',time_step)
        
        # Train/Fit and Test the Model
        regressor.fit(BP_X_train, BP_y_train, validation_data=(BP_X_test,BP_y_test), epochs=11, batch_size=64, verbose=1)
        train_predict=regressor.predict(BP_X_train)
        test_predict=regressor.predict(BP_X_test)
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)
        step=len(BP_test_data)-time_step
        x_input=BP_test_data[step:].reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        
        # Make prediction for the next 10 days
        lst_output=[]
        n_steps=time_step
        i=0
        days=int(Days.get())
        while(i<days):
            if(len(temp_input)>n_steps):
                x_input=np.array(temp_input[1:])
                #print("{} day input{}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print("{} day output{}".format(i,ythat))
                temp_input.extend(ythat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(ythat.tolist())
                i=i+1   
            else:
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print(ythat[0])
                temp_input.extend(ythat[0].tolist())
                lst_output.extend(ythat.tolist())
                i=i+1        
        
        # Visualization
        fig = plt.figure(figsize=(10,5))
        fig.suptitle('Stock Closing Price Prediction [BXPHARMA]', fontsize='12')
        data3=BP_data2.tolist()
        data3.extend(lst_output) 
        plt.plot(data3[:],color='green',alpha=0.9, label="Closing price of next 10 days [Prediction]")
        plt.plot(BP_data2[:],color='firebrick',alpha=0.9, label="Previous closing price [Train Data]")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.xlabel('Number of Days',fontsize='11.5')
        plt.ylabel('Stock Closing Price',fontsize='11.5')
        plt.show()
        if (days==5):
            fig.savefig('Output/BXPHARMA-5 days.jpg')
        else:
            fig.savefig('Output/BXPHARMA-10 days.jpg')
    
    else:
        print("Please select a company!")


def Compare():
    selection1=Select1.get()
    selection2=Select2.get()
    if (selection1=='BEXIMCO' or selection2=='BEXIMCO'):
        if (selection1=='LANKABANGLA' or selection2=='LANKABANGLA'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Market Volume Comparison', fontsize='12')
            plt.plot(BEXIMCO["VOLUME"],color='firebrick',alpha=0.9, label="BEXIMCO")
            plt.plot(LANKABANGLA["VOLUME"],color='blue',alpha=0.9, label="LANKABANGLA")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Market Volume',fontsize='11.5')
            plt.show()
        elif (selection1=='BATBC' or selection2=='BATBC'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Market Volume Comparison', fontsize='12')
            plt.plot(BEXIMCO["VOLUME"],color='firebrick',alpha=0.9, label="BEXIMCO")
            plt.plot(BATBC["VOLUME"],color='blue',alpha=0.9, label="BATBC")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Market Volume',fontsize='11.5')
            plt.show() 

        elif (selection1=='SAIFPOWER' or selection2=='SAIFPOWER'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Market Volume Comparison', fontsize='12')
            plt.plot(BEXIMCO["VOLUME"],color='firebrick',alpha=0.9, label="BEXIMCO")
            plt.plot(SAIFPOWER["VOLUME"],color='blue',alpha=0.9, label="SAIFPOWER")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Market Volume',fontsize='11.5')
            plt.show() 

        elif (selection1=='BXPHARMA' or selection2=='BXPHARMA'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Market Volume Comparison', fontsize='12')
            plt.plot(BEXIMCO["VOLUME"],color='firebrick',alpha=0.9, label="BEXIMCO")
            plt.plot(BXPHARMA["VOLUME"],color='blue',alpha=0.9, label="BXPHARMA")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Market Volume',fontsize='11.5')
            plt.show() 

    elif (selection1=='LANKABANGLA' or selection2=='LANKABANGLA'):
        if (selection1=='BATBC' or selection2=='BATBC'):
        # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Market Volume Comparison', fontsize='12')
            plt.plot(BATBC["VOLUME"],color='green',alpha=0.9, label="BATBC")
            plt.plot(LANKABANGLA["VOLUME"],color='blue',alpha=0.9, label="LANKABANGLA")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Market Volume',fontsize='11.5')
            plt.show()    
            
        elif (selection1=='BXPHARMA' or selection2=='BXPHARMA'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Market Volume Comparison', fontsize='12')
            plt.plot(LANKABANGLA["VOLUME"],color='firebrick',alpha=0.9, label="LANKABANGLA")
            plt.plot(BXPHARMA["VOLUME"],color='blue',alpha=0.9, label="BXPHARMA")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Market Volume',fontsize='11.5')
            plt.show() 

        elif (selection1=='SAIFPOWER' or selection2=='SAIFPOWER'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Market Volume Comparison', fontsize='12')
            plt.plot(LANKABANGLA["VOLUME"],color='firebrick',alpha=0.9, label="LANKABANGLA")
            plt.plot(SAIFPOWER["VOLUME"],color='blue',alpha=0.9, label="SAIFPOWER")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Market Volume',fontsize='11.5')
            plt.show() 
            
    elif (selection1=='BATBC' or selection2=='BATBC'):
        if (selection1=='SAIFPOWER' or selection2=='SAIFPOWER'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Market Volume Comparison', fontsize='12')
            plt.plot(BATBC["VOLUME"],color='firebrick',alpha=0.9, label="BATBC")
            plt.plot(SAIFPOWER["VOLUME"],color='blue',alpha=0.9, label="SAIFPOWER")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Market Volume',fontsize='11.5')
            plt.show()
        elif (selection1=='BXPHARMA' or selection2=='BXPHARMA'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Market Volume Comparison', fontsize='12')
            plt.plot(BATBC["VOLUME"],color='firebrick',alpha=0.9, label="BATBC")
            plt.plot(BXPHARMA["VOLUME"],color='blue',alpha=0.9, label="BXPHARMA")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Market Volume',fontsize='11.5')
            plt.show() 

    elif (selection1=='BXPHARMA' or selection2=='BXPHARMA'):
        if (selection1=='SAIFPOWER' or selection2=='SAIFPOWER'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Market Volume Comparison', fontsize='12')
            plt.plot(BXPHARMA["VOLUME"],color='firebrick',alpha=0.9, label="BXPHARMA")
            plt.plot(SAIFPOWER["VOLUME"],color='blue',alpha=0.9, label="SAIFPOWER")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Market Volume',fontsize='11.5')
            plt.show()        

    else:
        print("Please Select two different companies!")


def CLOSE():
    Property=Prop.get()
    if (Property=='BEXIMCO'):
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Closing Price [BEXIMCO]', fontsize='12')
            plt.plot(BEXIMCO["CLOSEP*"],color='firebrick',alpha=0.9, label="Closing Price")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('Stock Closing Price',fontsize='11.5')
            plt.show()
            
    elif (Property=='BATBC'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Closing Price [BATBC]', fontsize='12')
            plt.plot(BATBC["CLOSEP*"],color='green',alpha=0.9, label="Closing Price")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('Stock Closing Price',fontsize='11.5')
            plt.show()  

    elif (Property=='LANKABANGLA'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Closing Price [LANKABANGLA]', fontsize='12')
            plt.plot(LANKABANGLA["CLOSEP*"],color='blue',alpha=0.9, label="Closing Price")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('Stock Closing Price',fontsize='11.5')
            plt.show() 
            
    elif (Property=='SAIFPOWER'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Closing Price [SAIFPOWER]', fontsize='12')
            plt.plot(SAIFPOWER["CLOSEP*"],color='blue',alpha=0.9, label="Closing Price")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('Stock Closing Price',fontsize='11.5')
            plt.show()    
    
    elif (Property=='BXPHARMA'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Closing Price [BXPHARMA]', fontsize='12')
            plt.plot(BXPHARMA["CLOSEP*"],color='blue',alpha=0.9, label="Closing Price")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('Stock Closing Price',fontsize='11.5')
            plt.show()             
    
    else:
        print("Please Select a company!")        


def Trade():
    Property=Prop.get()
    if (Property=='BEXIMCO'):
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily Trade [BEXIMCO]', fontsize='12')
            plt.plot(BEXIMCO["TRADE"],color='firebrick',alpha=0.9, label="Daily Trade")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('Number of Trades',fontsize='11.5')
            plt.show()
    elif (Property=='BATBC'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily Trade [BATBC]', fontsize='12')
            plt.plot(BATBC["TRADE"],color='green',alpha=0.9, label="Daily Trade")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('Number of Trades',fontsize='11.5')
            plt.show()  

    elif (Property=='LANKABANGLA'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily Trade [LANKABANGLA]', fontsize='12')
            plt.plot(LANKABANGLA["TRADE"],color='blue',alpha=0.9, label="Daily Trade")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('Number of Trades',fontsize='11.5')
            plt.show()             
    
    elif (Property=='SAIFPOWER'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily Trade [SAIFPOWER]', fontsize='12')
            plt.plot(SAIFPOWER["TRADE"],color='blue',alpha=0.9, label="Closing Price")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('Stock Closing Price',fontsize='11.5')
            plt.show()    
    
    elif (Property=='BXPHARMA'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily Trade [BXPHARMA]', fontsize='12')
            plt.plot(BXPHARMA["TRADE"],color='blue',alpha=0.9, label="Closing Price")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('Stock Closing Price',fontsize='11.5')
            plt.show()             
    
    else:
        print("Please Select a company!")  
        
        
def HVL():
    Property=Prop.get()    
    if (Property=='BEXIMCO'):
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily High vs Low [BEXIMCO]', fontsize='12')
            plt.plot(BEXIMCO["HIGH"],color='Green',alpha=0.9, label="High")
            plt.plot(BEXIMCO["LOW"],color='firebrick',alpha=0.9, label="Low")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('High vs Low Price',fontsize='11.5')
            plt.show()
    elif (Property=='BATBC'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily High vs Low [BATBC]', fontsize='12')
            plt.plot(BATBC["HIGH"],color='Green',alpha=0.9, label="High")
            plt.plot(BATBC["LOW"],color='firebrick',alpha=0.9, label="Low")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('High vs Low Price',fontsize='11.5')
            plt.show()  

    elif (Property=='LANKABANGLA'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily High vs Low [LANKABANGLA]', fontsize='12')
            plt.plot(LANKABANGLA["HIGH"],color='Green',alpha=0.9, label="High")
            plt.plot(LANKABANGLA["LOW"],color='firebrick',alpha=0.9, label="Low")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('High vs Low Price',fontsize='11.5')
            plt.show()  
            
    elif (Property=='SAIFPOWER'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily High vs Low [SAIFPOWER]', fontsize='12')
            plt.plot(SAIFPOWER["HIGH"],color='Green',alpha=0.9, label="High")
            plt.plot(SAIFPOWER["LOW"],color='firebrick',alpha=0.9, label="Low")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('High vs Low Price',fontsize='11.5')
            plt.show()  

    elif (Property=='BXPHARMA'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily High vs Low [BXPHARMA]', fontsize='12')
            plt.plot(BXPHARMA["HIGH"],color='Green',alpha=0.9, label="High")
            plt.plot(BXPHARMA["LOW"],color='firebrick',alpha=0.9, label="Low")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Day',fontsize='11.5')
            plt.ylabel('High vs Low Price',fontsize='11.5')
            plt.show()  
            
    else:
        print("Please Select a company!")  


# GUI Part
root = Tk()
photo = PhotoImage(file = "E:\IIT\Final Project\Code\App\stock-market.png")
root.iconphoto(False, photo)
root.wm_title("StockApp")
root.configure(background='white')
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
OPTIONS = sorted(l1)
DAYS = pred_day


# Closing Price Prediction
#Header
w2 = Label(root, text="Predict Stock Closing Price", fg="midnight blue", bg="White")
w2.config(font=("Poppins",12,"bold"))
w2.grid(row=1, column=0, columnspan=4, pady=10)
#Options
Company = StringVar()
Company.set("Choose Company")
S1Lb = Label(root, text="Company Name:", fg="Black", bg="White")
S1Lb.config(font=("Poppins",10,"bold"))
S1Lb.grid(row=6, column=0, padx=15, pady=15, sticky=W)
S1 = OptionMenu(root, Company,*OPTIONS)
S1.grid(row=6, column=1)
#Date Options
Days = StringVar()
Days.set("10")
#S5Lb = Label(root, text="Number of Days:", fg="Black", bg="White")
#S5Lb.config(font=("Poppins",10,"bold"))
#S5Lb.grid(row=7, column=0, padx=15, pady=15, sticky=W)
#S1 = OptionMenu(root, Days,*DAYS)
#S1.grid(row=7, column=1)
#Button
lstm = Button(root, text="Predict", command=LSTM,bg="midnight blue",fg="white")
lstm.config(font=("poppins",10,"bold"))
lstm.grid(row=6, column=2, padx=15, pady=15)


# Stock Properties Visualization
#Header
w1 = Label(root, text="Visualize Stock Properties", fg="midnight blue", bg="White")
w1.config(font=("Poppins",12,"bold"))
w1.grid(row=9, column=0, columnspan=4, pady=10)
#Options
Prop = StringVar()
Prop.set("Choose Company")
S4Lb = Label(root, text="Company Name:", fg="Black", bg="White")
S4Lb.config(font=("Poppins",10,"bold"))
S4Lb.grid(row=10, column=0, padx=15, pady=15, sticky=W)
S4 = OptionMenu(root, Prop,*OPTIONS)
S4.grid(row=10, column=1) 
#Buttons
CLOSE = Button(root, text="Closing Price", command=CLOSE,bg="midnight blue",fg="white")
CLOSE.config(font=("poppins",10,"bold"))
CLOSE.grid(row=11, column=0, padx=10, pady=15)
hvl = Button(root, text="High vs Low", command=HVL,bg="midnight blue",fg="white")
hvl.config(font=("poppins",10,"bold"))
hvl.grid(row=11, column=1, padx=10, pady=15)
trd = Button(root, text="Daily Trade", command=Trade,bg="midnight blue",fg="white")
trd.config(font=("poppins",10,"bold"))
trd.grid(row=11, column=2, padx=10, pady=15)
#Divider
w3 = Label(root, text="___________________________________________", fg="midnight blue", bg="White")
w3.config(font=("Poppins",12,"bold"))
w3.grid(row=12, column=0, columnspan=4, pady=10)
w2 = Label(root, text="___________________________________________", fg="midnight blue", bg="White")
w2.config(font=("Poppins",12,"bold"))
w2.grid(row=8, column=0, columnspan=4, pady=10)


# Market Volume Comparison
#Header
w3 = Label(root, text="Compare Market Volume", fg="midnight blue", bg="White")
w3.config(font=("Poppins",12,"bold"))
w3.grid(row=13, column=0, columnspan=4, pady=10)
#Options
Select1 = StringVar()
Select1.set("Choose Company")
Select2 = StringVar()
Select2.set("Choose Company")
S2Lb = Label(root, text="Company Name [1]:", fg="Black", bg="White")
S2Lb.config(font=("Poppins",10,"bold"))
S2Lb.grid(row=14, column=0, padx=15, pady=15, sticky=W)
S3Lb = Label(root, text="Company Name [2]:", fg="Black", bg="White")
S3Lb.config(font=("Poppins",10,"bold"))
S3Lb.grid(row=15, column=0, padx=15, pady=15, sticky=W)
S2 = OptionMenu(root, Select1,*OPTIONS)
S2.grid(row=14, column=1) 
S3 = OptionMenu(root, Select2,*OPTIONS)
S3.grid(row=15, column=1) 
#Button
cmp = Button(root, text="Compare", command=Compare,bg="midnight blue",fg="white")
cmp.config(font=("poppins",10,"bold"))
cmp.grid(row=14, column=2, padx=15, pady=15)


#Credit
w4 = Label(root, text="Kamrul Hasan | MIT 21st Batch", fg="White", bg="midnight blue")
w4.config(font=("Poppins",8,"bold"))
w4.grid(row=16, column=0, columnspan=4, pady=15)
root.mainloop()