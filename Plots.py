# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:11:08 2022

@author: Will Pc
"""

import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np


Files = [r'C:\Users\Will Pc\iCloudDrive\Math\Math3599\IndexFutures\ContractFutures_1min_uic4039.csv', r'C:\Users\Will Pc\iCloudDrive\Math\Math3599\IndexFutures\ContractFutures_1min_uic4044.csv', r'C:\Users\Will Pc\iCloudDrive\Math\Math3599\IndexFutures\ContractFutures_1min_uic9046.csv']
fl = len(Files)
Titles = ['S&P 500', 'FTSE 100', 'ASX 200']
for i in range(fl):
    Data_df = pd.read_csv(Files[i])
    Data_df['Time']= pd.to_datetime(Data_df['Time'])
    Data_df['Time'] = Data_df['Time'].dt.date
    Data_df.set_index("Time", inplace=True) #sets the index of the dataframe by date
    plt.figure()
    Data_df.Close.plot(label = 'Close')
    plt.ylabel("Close")
    plt.title('Close Data of ' + (Titles[i]))
    plt.savefig(f"close_{Titles[i]}.png", format="PNG")
    plt.close
    #L = np.percentile(data, 100-confidence)
    #U = np.percentile(data, confidence)
    #M = np.mean(data)
    #Plot = data.plot(label = 'Observed') 
    #Plot.axhline(y=U,c='g', label = 'Percentile band')
    #Plot.axhline(y=L,c='g')
    #Plot.axhline(y=M,c='r', label = 'Average')
    #Plot.set_ylabel('Close Difference')
    #Plot.legend(loc = "lower right")
    #Plot.legend()