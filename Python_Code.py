#----------------------------------------------------------
# Import Relevant Libraries
#----------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

#----------------------------------------------------------
# Select & Import Correct Indicies
#----------------------------------------------------------

# E-mini S&P 500 (Dollar) - continuous
data = pd.read_csv (r'C:\Users\meht1\OneDrive\Desktop\MATH3599\Index_Analysis\IndexFutures\IndexFutures\ContractFutures_1min_uic4039.csv')   

# FTSE 100 Index - continuous
# data = pd.read_csv (r'C:\Users\lenno\OneDrive\Desktop\S2 2022\MATH3599\Resources\IndexFutures\IndexFutures\ContractFutures_1min_uic4044.csv')   

#ASX SPI 200 Index - continuous
# data = pd.read_csv (r'C:\Users\lenno\OneDrive\Desktop\S2 2022\MATH3599\Resources\IndexFutures\IndexFutures\ContractFutures_1min_uic9046.csv')   


#----------------------------------------------------------
# Clean & Set-Up Data for Classification
#----------------------------------------------------------

hist_qoutes = pd.DataFrame(data, columns= ['Item','Close','High','Interest','Low','Open','Time','Volume'])

#clean date to match time:
    # 2016-12-30T02:34:00.000000Z - IN OUR DATA
    # 2001-01-02 00:00:00 - DESIRED
hist_qoutes['Time'] = hist_qoutes['Time'].str.replace('T',' ')
hist_qoutes['Time'] = hist_qoutes['Time'].str.replace('.000000Z',' ')

#Remove Item Col
hist_qoutes.drop('Item', inplace=True, axis=1)
hist_qoutes.drop('Interest', inplace=True, axis=1)

#Rename 'Time' to 'Date'
hist_qoutes.rename(columns = {'Time':'Date'}, inplace = True)


# print(hist_qoutes.columns.tolist())
print(hist_qoutes.head())

plt.plot(hist_qoutes['Close'])
plt.xlabel('Time Minutes')
plt.ylabel('Close Price')



# add the outcome variable, 1 if the trading session was positive (close>open), 0 otherwise
hist_qoutes['Outcome'] = hist_qoutes.apply(lambda x: 1 if x['Close'] > x['Open'] else -1)
# we also need to add three new columns ‘ho’ ‘lo’ and ‘gain’
# they will be useful to backtest the model, later
hist_qoutes['Ho'] = hist_qoutes['High'] - hist_qoutes['Open'] # distance between Highest and Opening price
hist_qoutes['Lo'] = hist_qoutes['Low'] - hist_qoutes['Open'] # distance between Lowest and Opening price
hist_qoutes['Gain'] = hist_qoutes['Close'] - hist_qoutes['Open']

