from sklearn.preprocessing  import MinMaxScaler;
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
from keras.models import Sequential;
from keras.layers import Dense;
from keras.layers import LSTM;
from tkinter import *

# BEXIMCO

# Load Data and Convert Data Type
BEXIMCO_data=pd.read_csv(r'data/BEXIMCO.csv', index_col="DATE", parse_dates=True)
#BEXIMCO_data["OPEN"] = BEXIMCO_data["OPEN"].str.replace(',', '').astype(float);
#BEXIMCO_data["HIGH"] = BEXIMCO_data["HIGH"].str.replace(',', '').astype(float);
#BEXIMCO_data["LOW"] = BEXIMCO_data["LOW"].str.replace(',', '').astype(float);
#BEXIMCO_data["CLOSE"] = BEXIMCO_data["CLOSE"].str.replace(',', '').astype(float);
BEXIMCO_data["VOLUME"] = BEXIMCO_data["VOLUME"].str.replace(',', '').astype(float);
BEXIMCO_data["TRADE"] = BEXIMCO_data["TRADE"].str.replace(',', '').astype(float);

BEXIMCO_data1 = BEXIMCO_data.reset_index()['CLOSE'];
#print(BEXIMCO_data.head())

# Convert/Scale the Data
scaler= MinMaxScaler(feature_range = (0,1))
BEXIMCO_data2 = scaler.fit_transform(np.array(BEXIMCO_data1).reshape(-1,1));

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
BATBC_data["HIGH"] = BATBC_data["HIGH"].str.replace(',', '').astype(float);
BATBC_data["LOW"] = BATBC_data["LOW"].str.replace(',', '').astype(float);
BATBC_data["CLOSE"] = BATBC_data["CLOSE"].str.replace(',', '').astype(float);
BATBC_data["VOLUME"] = BATBC_data["VOLUME"].str.replace(',', '').astype(float);
BATBC_data["TRADE"] = BATBC_data["TRADE"].str.replace(',', '').astype(float);
BATBC_data1 = BATBC_data.reset_index()['CLOSE'];
#print(BATBC_data.info())

# Convert/Scale the Data
scaler= MinMaxScaler(feature_range = (0,1))
BATBC_data2 = scaler.fit_transform(np.array(BATBC_data1).reshape(-1,1));

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
LB_data["TRADE"] = LB_data["TRADE"].str.replace(',', '').astype(float);
LB_data["VOLUME"] = LB_data["VOLUME"].str.replace(',', '').astype(float);
LB_data1 = LB_data.reset_index()['CLOSE'];
#print(LB_data.info())

# Convert/Scale the Data
scaler= MinMaxScaler(feature_range = (0,1))
LB_data2 = scaler.fit_transform(np.array(LB_data1).reshape(-1,1));

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

# 1st LSTM Layer
regressor.add(LSTM(50, return_sequences=True, input_shape = (time_step,1)));

# 2nd LSTM Layer
regressor.add(LSTM(50, return_sequences=True));

# 3rd LSTM Layer
regressor.add(LSTM(50))

regressor.add(Dense(1))
regressor.compile(loss='mean_squared_error', optimizer='adam');

l1=['BEXIMCO', 'BATBC', 'LANKABANGLA']
pred_day = ['5','10']

def LSTM():
    selection=Company.get()
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
        step=len(BEX_test_data)-time_step
        x_input=BEX_test_data[step:].reshape(1,-1);
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist();
        
        # Make prediction for the next 10 days
        lst_output=[]
        n_steps=time_step;
        i=0
        days=int(Days.get())
        while(i<days):
            if(len(temp_input)>n_steps):
                x_input=np.array(temp_input[1:])
                #print("{} day input{}".format(i,x_input));
                x_input=x_input.reshape(1,-1);
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print("{} day output{}".format(i,ythat));
                temp_input.extend(ythat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(ythat.tolist())
                i=i+1;   
            else:
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print(ythat[0])
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
        step=len(BAT_test_data)-time_step;
        x_input=BAT_test_data[step:].reshape(1,-1);
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist();
        
        # Make prediction for the next 10 days
        lst_output=[]
        n_steps=time_step;
        i=0
        days=int(Days.get())
        while(i<days):
            if(len(temp_input)>n_steps):
                x_input=np.array(temp_input[1:])
                #print("{} day input{}".format(i,x_input));
                x_input=x_input.reshape(1,-1);
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print("{} day output{}".format(i,ythat));
                temp_input.extend(ythat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(ythat.tolist())
                i=i+1;   
            else:
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print(ythat[0])
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
        step=len(LB_test_data)-time_step
        x_input=LB_test_data[step:].reshape(1,-1);
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist();
        
        # Make prediction for the next 10 days
        lst_output=[]
        n_steps=time_step;
        i=0
        days=int(Days.get())
        while(i<days):
            if(len(temp_input)>n_steps):
                x_input=np.array(temp_input[1:])
                #print("{} day input{}".format(i,x_input));
                x_input=x_input.reshape(1,-1);
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print("{} day output{}".format(i,ythat));
                temp_input.extend(ythat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(ythat.tolist())
                i=i+1;   
            else:
                x_input=x_input.reshape((1, n_steps,1))
                ythat=regressor.predict(x_input, verbose=0)
                #print(ythat[0])
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
            plt.plot(BEXIMCO_data["VOLUME"],color='firebrick',alpha=0.9, label="BEXIMCO")
            plt.plot(LB_data["VOLUME"],color='blue',alpha=0.9, label="LANKABANGLA")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Market Volume',fontsize='11.5')
            plt.show()
        elif (selection1=='BATBC' or selection2=='BATBC'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Market Volume Comparison', fontsize='12')
            plt.plot(BEXIMCO_data["VOLUME"],color='firebrick',alpha=0.9, label="BEXIMCO")
            plt.plot(BATBC_data["VOLUME"],color='blue',alpha=0.9, label="BATBC")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Market Volume',fontsize='11.5')
            plt.show()  

    elif (selection1=='LANKABANGLA' or selection2=='LANKABANGLA'):
        if (selection1=='BATBC' or selection2=='BATBC'):
        # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Market Volume Comparison', fontsize='12')
            plt.plot(BATBC_data["VOLUME"],color='green',alpha=0.9, label="BATBC")
            plt.plot(LB_data["VOLUME"],color='blue',alpha=0.9, label="LANKABANGLA")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Market Volume',fontsize='11.5')
            plt.show()             
    else:
        print("Please Select two different companies!")
        
def Close():
    Property=Prop.get()
    
    if (Property=='BEXIMCO'):
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Closing Price [BEXIMCO]', fontsize='12')
            plt.plot(BEXIMCO_data["CLOSE"],color='firebrick',alpha=0.9, label="Closing Price")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Stock Closing Price',fontsize='11.5')
            plt.show()
    elif (Property=='BATBC'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Closing Price [BATBC]', fontsize='12')
            plt.plot(BATBC_data["CLOSE"],color='green',alpha=0.9, label="Closing Price")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Stock Closing Price',fontsize='11.5')
            plt.show()  

    elif (Property=='LANKABANGLA'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Closing Price [LANKABANGLA]', fontsize='12')
            plt.plot(LB_data["CLOSE"],color='blue',alpha=0.9, label="Closing Price")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Stock Closing Price',fontsize='11.5')
            plt.show()             
    else:
        print("Please Select a company!")        

def Trade():
    Property=Prop.get()
    
    if (Property=='BEXIMCO'):
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily Trade [BEXIMCO] ', fontsize='12')
            plt.plot(BEXIMCO_data["TRADE"],color='firebrick',alpha=0.9, label="Daily Trade")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Number of Trades',fontsize='11.5')
            plt.show()
    elif (Property=='BATBC'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily Trade [BATBC]', fontsize='12')
            plt.plot(BATBC_data["TRADE"],color='green',alpha=0.9, label="Daily Trade")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Number of Trades',fontsize='11.5')
            plt.show()  

    elif (Property=='LANKABANGLA'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily Trade [LANKABANGLA]', fontsize='12')
            plt.plot(LB_data["TRADE"],color='blue',alpha=0.9, label="Daily Trade")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Number of Trades',fontsize='11.5')
            plt.show()             
    else:
        print("Please Select a company!")  
        
def HVL():
    Property=Prop.get()
    
    if (Property=='BEXIMCO'):
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily High vs Low [BEXIMCO]', fontsize='12')
            plt.plot(BEXIMCO_data["HIGH"],color='Green',alpha=0.9, label="High")
            plt.plot(BEXIMCO_data["LOW"],color='firebrick',alpha=0.9, label="Low")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('High vs Low Price',fontsize='11.5')
            plt.show()
    elif (Property=='BATBC'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily High vs Low [BATBC]', fontsize='12')
            plt.plot(BATBC_data["HIGH"],color='Green',alpha=0.9, label="High")
            plt.plot(BATBC_data["LOW"],color='firebrick',alpha=0.9, label="Low")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('High vs Low Price',fontsize='11.5')
            plt.show()  

    elif (Property=='LANKABANGLA'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Daily High vs Low [LANKABANGLA]', fontsize='12')
            plt.plot(LB_data["HIGH"],color='Green',alpha=0.9, label="High")
            plt.plot(LB_data["LOW"],color='firebrick',alpha=0.9, label="Low")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('High vs Low Price',fontsize='11.5')
            plt.show()             
    else:
        print("Please Select a company!")  


# GUI Part
root = Tk()
root.wm_title("StockApp")
root.configure(background='white')
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
OPTIONS = sorted(l1)
DAYS = pred_day
# Prediction

w2 = Label(root, text="Predict Stock Closing Price", fg="midnight blue", bg="White")
w2.config(font=("Poppins",12,"bold"))
w2.grid(row=1, column=0, columnspan=4, pady=10)
Company = StringVar()
Company.set("Choose Company")
S1Lb = Label(root, text="Company Name:", fg="Black", bg="White")
S1Lb.config(font=("Poppins",10,"bold"))
S1Lb.grid(row=6, column=0, padx=15, pady=15, sticky=W)
S1 = OptionMenu(root, Company,*OPTIONS)
S1.grid(row=6, column=1)
Days = StringVar()
Days.set("10")
S5Lb = Label(root, text="Number of Days:", fg="Black", bg="White")
S5Lb.config(font=("Poppins",10,"bold"))
S5Lb.grid(row=7, column=0, padx=15, pady=15, sticky=W)
S1 = OptionMenu(root, Days,*DAYS)
S1.grid(row=7, column=1)
lstm = Button(root, text="Predict", command=LSTM,bg="midnight blue",fg="white")
lstm.config(font=("poppins",10,"bold"))
lstm.grid(row=6, column=2, padx=15, pady=15)

w6 = Label(root, text="___________________________________________", fg="midnight blue", bg="White")
w6.config(font=("Poppins",12,"bold"))
w6.grid(row=8, column=0, columnspan=4, pady=10)

# Stock Properties
w5 = Label(root, text="Visualize Stock Properties", fg="midnight blue", bg="White")
w5.config(font=("Poppins",12,"bold"))
w5.grid(row=9, column=0, columnspan=4, pady=10)
Prop = StringVar()
Prop.set("Choose Company")
S4Lb = Label(root, text="Company Name:", fg="Black", bg="White")
S4Lb.config(font=("Poppins",10,"bold"))
S4Lb.grid(row=10, column=0, padx=15, pady=15, sticky=W)
S4 = OptionMenu(root, Prop,*OPTIONS)
S4.grid(row=10, column=1) 
close = Button(root, text="Close Price", command=Close,bg="midnight blue",fg="white")
close.config(font=("poppins",10,"bold"))
close.grid(row=11, column=0, padx=10, pady=15)
hvl = Button(root, text="High vs Low", command=HVL,bg="midnight blue",fg="white")
hvl.config(font=("poppins",10,"bold"))
hvl.grid(row=11, column=1, padx=10, pady=15)
trd = Button(root, text="Daily Trade", command=Trade,bg="midnight blue",fg="white")
trd.config(font=("poppins",10,"bold"))
trd.grid(row=11, column=2, padx=10, pady=15)

w6 = Label(root, text="___________________________________________", fg="midnight blue", bg="White")
w6.config(font=("Poppins",12,"bold"))
w6.grid(row=12, column=0, columnspan=4, pady=10)

# Comparison
w3 = Label(root, text="Compare Market Volume", fg="midnight blue", bg="White")
w3.config(font=("Poppins",12,"bold"))
w3.grid(row=13, column=0, columnspan=4, pady=10)
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
cmp = Button(root, text="Compare", command=Compare,bg="midnight blue",fg="white")
cmp.config(font=("poppins",10,"bold"))
cmp.grid(row=14, column=2, padx=15, pady=15)
w4 = Label(root, text="Kamrul Hasan | MIT 21st Batch", fg="White", bg="midnight blue")
w4.config(font=("Poppins",8,"bold"))
w4.grid(row=16, column=0, columnspan=4, pady=15)
root.mainloop()