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

# ---------------------------------------------------------------------------------------

l1=['BEXIMCO', 'BATBC', 'LANKABANGLA']

def Compare():
    selection1=Symptom1.get()
    selection2=Symptom2.get()
    
    if (selection1=='BEXIMCO' or selection2=='BEXIMCO'):
        if (selection1=='LANKABANGLA' or selection2=='LANKABANGLA'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Closing Price Comparison', fontsize='12')
            plt.plot(BEXIMCO_data2[100:],color='firebrick',alpha=0.9, label="BEXIMCO")
            plt.plot(LB_data2[100:],color='blue',alpha=0.9, label="LANKABANGLA")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Stock Closing Price',fontsize='11.5')
            plt.show()
        elif (selection1=='BATBC' or selection2=='BATBC'):
            # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Closing Price Comparison', fontsize='12')
            plt.plot(BEXIMCO_data2[100:],color='firebrick',alpha=0.9, label="BEXIMCO")
            plt.plot(BATBC_data2[100:],color='blue',alpha=0.9, label="BATBC")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Stock Closing Price',fontsize='11.5')
            plt.show()  

    elif (selection1=='LANKABANGLA' or selection2=='LANKABANGLA'):
        if (selection1=='BATBC' or selection2=='BATBC'):
        # Visualization
            fig = plt.figure(figsize=(10,5))
            fig.suptitle('Stock Closing Price Comparison', fontsize='12')
            plt.plot(BATBC_data2[100:],color='green',alpha=0.9, label="BATBC")
            plt.plot(LB_data2[100:],color='blue',alpha=0.9, label="LANKABANGLA")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel('Time',fontsize='11.5')
            plt.ylabel('Stock Closing Price',fontsize='11.5')
            plt.show()
              
    else:
        print("Please Select two different companies!")

# GUI Part
root = Tk()
root.wm_title("Stock Closing Price Comparison")
root.configure(background='white')
w2 = Label(root, text="Compare Stock Closing Prices", fg="Orange", bg="White")
w2.config(font=("Poppins",16,"bold"))
w2.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
Symptom1 = StringVar()
Symptom1.set("Choose Company")
Symptom2 = StringVar()
Symptom2.set("Choose Company")
S1Lb = Label(root, text="First Company Name:", fg="Black", bg="White")
S1Lb.config(font=("Poppins",10,"bold"))
S1Lb.grid(row=7, column=0, padx=10, pady=10, sticky=W)
S2Lb = Label(root, text="Second Company Name:", fg="Black", bg="White")
S2Lb.config(font=("Poppins",10,"bold"))
S2Lb.grid(row=8, column=0, padx=10, pady=10, sticky=W)
OPTIONS = sorted(l1)
S1 = OptionMenu(root, Symptom1,*OPTIONS)
S1.grid(row=7, column=1) 
S2 = OptionMenu(root, Symptom2,*OPTIONS)
S2.grid(row=8, column=1) 
lstm = Button(root, text="Compare", command=Compare,bg="Orange",fg="white")
lstm.config(font=("poppins",10,"bold"))
lstm.grid(row=7, column=3, padx=10, pady=10)
root.mainloop()