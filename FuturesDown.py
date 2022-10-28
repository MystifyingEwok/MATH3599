# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 15:22:24 2022

@author: Will Pc
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 13:00:50 2022

@author: Will Pc
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 01:31:08 2022

@author: Will Pc
"""

import yfinance as yf
import matplotlib as mp
import datetime as dt
import pandas as pd
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime

 
 #Data_df['inward_date'] = Data_df['Time'].apply(lambda x: datetime.strptime(x[:10], "%Y-%m-%d").strftime("%d-%m-%Y"))
 
 #Importing the data, would need to change the directory but works at the moment,
#Data_df = pd.read_csv(r'C:\Users\Will Pc\iCloudDrive\Math\Math3599\IndexFutures\ContractFutures_1min_uic9046.csv')
DF= []
Files = [r'C:\Users\Will Pc\iCloudDrive\Math\Math3599\IndexFutures\ContractFutures_1min_uic4039.csv', r'C:\Users\Will Pc\iCloudDrive\Math\Math3599\IndexFutures\ContractFutures_1min_uic4044.csv', r'C:\Users\Will Pc\iCloudDrive\Math\Math3599\IndexFutures\ContractFutures_1min_uic9046.csv']
fl = len(Files)
for i in range(fl):
#Importing and manipulating the datafiles 
    Data_df = pd.read_csv(Files[i])
    Data_df['Time']= pd.to_datetime(Data_df['Time'])
    Data_df['Time'] = Data_df['Time'].dt.time
    Data_df.set_index("Time", inplace=True) #sets the index of the dataframe by date 
    
    #Using 20% of the file Size
    X = round(0.05* len(Data_df['Close']))
    Data_df = Data_df.tail(X)
    #Data_df = pd.read_csv(r'C:\Users\Will Pc\iCloudDrive\Math\Math3599\Test2.csv')
    Headers = list(Data_df)#Extracts the headers from the dataframe
     #Removes the null columns
     #Data_df.drop(['bid_volume', 'bid_average','bid_barCount', 'ask_volume', 'ask_average','ask_barCount'], axis =1, inplace=True)
    Data_df.drop('Unnamed: 0', axis = 1, inplace = True)
    #Extracts the Closing Price Column 
    data = Data_df[["Close"]]
    #This will also be the data we compare our results to.
    data = data.rename(columns = {'Close':'Actual_Close'})
    
    
    
    #This identifies if the price went up or down
    #data["Target"] = Data_df.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
    data["Target"] = Data_df.rolling(2).apply(lambda x: x.iloc[1] < x.iloc[0])["Close"]
    
    
    
    
    # Shift stock prices forward one day, so predict tomorrow prices from today prices.
    Data_prev = Data_df.copy()
    Data_prev = Data_prev.shift(1) 
    Data_prev.head()
    
    data = data[1:];
    
    # Create our training data
    predictors = ["Close", "Volume", "Open", "High", "Low", "Interest"]
    #data = data.join(Data_prev[predictors].iloc[1:], how = 'outer')
    X = Data_prev[predictors].iloc[1:]
    
    data = pd.concat([data, X], axis = 1)
    data.head()
    
    
    from sklearn.ensemble import RandomForestClassifier
    
    # Create a random forest classification model.  Set min_samples_split high to ensure we don't overfit.
    model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)
    
    # Create a train and test set
    train = data.iloc[:round(0.7*len(data)),:]
    test = data.iloc[round(0.7*len(data)):,:]
    data['Target'].value_counts()
    model.fit(train[predictors], train["Target"])
    
    
    from sklearn.metrics import precision_score
    
    # Evaluate error of predictions
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
    precision_score(test["Target"], preds)
    
    combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
    combined.plot()
    
    
    Accuracy = combined['Target'] - combined['Predictions']
    Y = Accuracy.value_counts()
    Incs = combined.loc[combined['Target'] == 1]; 
    Nincs = combined.loc[combined['Target'] == 0]; 
    NincAccuracy = Nincs['Target'] - Nincs['Predictions']
    NincAccuracyCounts = NincAccuracy.value_counts()
    IncAccuracy = Incs['Target'] - Incs['Predictions']
    IncAccuracyCounts = IncAccuracy.value_counts()
    Labels = ['Correct Increase', 'Missed Increase', 'Correct NoIncrease', 'Incorrect NoIncrease'];
    AccuracyData = [IncAccuracyCounts[0.0],IncAccuracyCounts[1.0],NincAccuracyCounts[0.0],NincAccuracyCounts[-1.0]];
    AccuracyData = pd.DataFrame(AccuracyData, Labels)
    DF = np.append(DF, AccuracyData)

Titles = ['E-mini S&P 500 Correct Increase ', 'E-mini S&P 500 Missed Increase', 'E-mini S&P 500 Correct NoIncrease', 'E-mini S&P 500 Incorrect NoIncrease','FTSE 100 Correct Increase', 'FTSE 100 Missed Increase', 'FTSE 100 Correct NoIncrease', 'FTSE 100 Incorrect NoIncrease','ASX SPI 200 Correct Increase', 'ASX SPI 200 Missed Increase', 'ASX SPI 200 Correct NoIncrease', 'ASX SPI 200 Incorrect NoIncrease'];
Results = pd.DataFrame(DF,Titles)