# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:34:07 2022

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
Data_df = pd.read_csv(r'C:\Users\Will Pc\iCloudDrive\Math\Math3599\IndexFutures\ContractFutures_1min_uic4039.csv')
Data_df['Time']= pd.to_datetime(Data_df['Time'])
Data_df['Time'] = Data_df['Time'].dt.time
Data_df.set_index("Time", inplace=True) #sets the index of the dataframe by date 
#X = round(0.1* len(Data_df['Close']))
#Data_df = Data_df.tail(X)
 #Data_df = pd.read_csv(r'C:\Users\Will Pc\iCloudDrive\Math\Math3599\Test2.csv')
Headers = list(Data_df)#Extracts the headers from the dataframe
 #Removes the null columns
 #Data_df.drop(['bid_volume', 'bid_average','bid_barCount', 'ask_volume', 'ask_average','ask_barCount'], axis =1, inplace=True)
Data_df.drop('Unnamed: 0', axis = 1, inplace = True)
data = Data_df[["Close"]]
data = data.rename(columns = {'Close':'Actual_Close'})


# Ensure we know the actual closing price

# Setup our target.  This identifies if the price went up or down
data["Target"] = Data_df.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]




# Shift stock prices forward one day, so we're predicting tomorrow's stock prices from today's prices.
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
train = data.iloc[:-100000]
test = data.iloc[-100000:]
data['Target'].value_counts()
model.fit(train[predictors], train["Target"])


from sklearn.metrics import precision_score

# Evaluate error of predictions
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
precision_score(test["Target"], preds)

combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
combined.plot()



i = 1000
step = 750

train = data.iloc[0:i].copy()
test = data.iloc[i:(i+step)].copy()
model.fit(train[predictors], train["Target"])
preds = model.predict(test[predictors])

preds = model.predict_proba(test[predictors])[:,1]
preds = pd.Series(preds, index=test.index)
preds[preds > .6] = 1
preds[preds<=.6] = 0

preds.head()

predictions = []
# Loop over the dataset in increments
for i in range(1000, data.shape[0], step):
    # Split into train and test sets
    train = data.iloc[0:i].copy()
    test = data.iloc[i:(i+step)].copy()

    # Fit the random forest model
model.fit(train[predictors], train["Target"])

    # Make predictions
preds = model.predict_proba(test[predictors])[:,1]
preds = pd.Series(preds, index=test.index)
preds[preds > .6] = 1
preds[preds<=.6] = 0

    # Combine predictions and test values
combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

predictions.append(combined)
    
predictions[0].head()
    
    
    
    
def backtest(data, model, predictors, start=1000, step=750):
        predictions = []
    # Loop over the dataset in increments
        for i in range(start, data.shape[0], step):
        # Split into train and test sets
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()

        # Fit the random forest model
        model.fit(train[predictors], train["Target"])

        # Make predictions
        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > .6] = 1
        preds[preds<=.6] = 0

        # Combine predictions and test values
        combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

        predictions.append(combined)

        return pd.concat(predictions)

predictions = backtest(data, model, predictors)

predictions["Predictions"].value_counts()

predictions["Target"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])

weekly_mean = data.rolling(7).mean()["Close"]
quarterly_mean = data.rolling(90).mean()["Close"]
annual_mean = data.rolling(365).mean()["Close"]

weekly_trend = data.shift(1).rolling(7).sum()["Target"]


data["weekly_mean"] = weekly_mean / data["Close"]
data["quarterly_mean"] = quarterly_mean / data["Close"]
data["annual_mean"] = annual_mean / data["Close"]

data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]

data["weekly_trend"] = weekly_trend


data["open_close_ratio"] = data["Open"] / data["Close"]
data["high_close_ratio"] = data["High"] / data["Close"]
data["low_close_ratio"] = data["Low"] / data["Close"]

full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high"]
                                
predictions = backtest(data.iloc[365:], model, full_predictors)  

precision_score(predictions["Target"], predictions["Predictions"])       
# Show how many trades we would make
predictions["Predictions"].value_counts()


# Look at trades we would have made in the last 100 days

predictions.iloc[-100:].plot()
print(combined['Target'].value_counts())
print(combined['Predictions'].value_counts())
                       