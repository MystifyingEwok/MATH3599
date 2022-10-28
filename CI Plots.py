# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 19:22:24 2022

@author: Will Pc
"""



import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np

Files = [r'C:\Users\Will Pc\iCloudDrive\Math\Math3599\IndexFutures\ContractFutures_1min_uic4039.csv', r'C:\Users\Will Pc\iCloudDrive\Math\Math3599\IndexFutures\ContractFutures_1min_uic4044.csv', r'C:\Users\Will Pc\iCloudDrive\Math\Math3599\IndexFutures\ContractFutures_1min_uic9046.csv']
   
Data_df = pd.read_csv(Files[1])
Data_df['Time']= pd.to_datetime(Data_df['Time'])
Data_df['Time'] = Data_df['Time'].dt.time
Data_df.set_index("Time", inplace=True) #sets the index of the dataframe by date 
data = Data_df['Close']



def ArrCretor(x):
    import random as rd 
    Len = len(data)
    Rand = rd.randrange(Len-x)
    Array = list(range(Rand,Rand+x))
    return Array




Indx = ArrCretor(100);
TSData = data[Indx];

from statsmodels.tsa.stattools import adfuller 
TestTS = adfuller(TSData)
 
#To view that Raw Data is not stationary, so try differencing
print('ADF Statistic: %f' % TestTS[0])
print('p-value: %f' % TestTS[1])
print('Critical Values:')
for key, value in TestTS[4].items():
	print('\t%s: %.3f' % (key, value))
    
StatTSData = TSData.diff(1);
StatTSData = StatTSData[1:]
STest = adfuller(StatTSData)

     
#To Verify Differenced Data is stationary 
print('ADF Statistic: %f' % STest[0])
print('p-value: %f' % STest[1])
print('Critical Values:')
for key, value in STest[4].items():
  print('\t%s: %.3f' % (key, value))

pred_matrix

    
#Do the Graphs on Differenced Data, therefore can make claims about this 'differencing' vs absolute prices


    # def mean_confidence_interval(data, confidence=0.95):
    #    import scipy.stats
   #     a = 1.0 * np.array(data)
    #    n = len(a)
    #    m, se = np.mean(a), scipy.stats.sem(a)
    #    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
     #   return m, m-h, m+h
     
def ciPlot(data,confidence = 90):
    L = np.percentile(data, 100-confidence)
    U = np.percentile(data, confidence)
    M = np.mean(data)
    Plot = data.plot(label = 'Observed') 
    Plot.axhline(y=U,c='g', label = 'Percentile band')
    Plot.axhline(y=L,c='g')
    Plot.axhline(y=M,c='r', label = 'Average')
    Plot.set_ylabel('Close Difference')
    Plot.legend(loc = "lower right")
    Plot.legend()
    
ciPlot(StatTSData, 0.95)