
#import graphlab as gl
#from __future__ import division
# from datetime import datetime
#import pandas as pd
#from yahoo_finance import Share

# download historical prices of S&P 500 index
# today = datetime.strftime(datetime.today(), "%Y-%m-%d")
# stock = Share('^GSPC') # ^GSPC is the Yahoo finance symbol to refer S&P 500 index
# we gather historical quotes from 2001-01-01 up to today
# hist_quotes = stock.get_historical('2001-01-01', today)
# here is how a row looks like
# hist_quotes[0]

# l_date = []
# l_open = []
# l_high = []
# l_low = []
# l_close = []
# l_volume = []
# # reverse the list
# hist_quotes.reverse()
# for quotes in hist_quotes:
#     l_date.append(quotes['Date'])
#     l_open.append(float(quotes['Open']))
#     l_high.append(float(quotes['High']))
#     l_low.append(float(quotes['Low']))
#     l_close.append(float(quotes['Close']))
#     l_volume.append(int(quotes['Volume']))
    
# qq = gl.SFrame({'datetime' : l_date, 
#           'open' : l_open, 
#           'high' : l_high, 
#           'low' : l_low, 
#           'close' : l_close, 
#           'volume' : l_volume})
# # datetime is a string, so convert into datetime object
# qq['datetime'] = qq['datetime'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

# # just to check if data is sorted in ascending mode
# qq.head(3)

# qq.save(“SP500_daily.bin”)
# # once data is saved, we can use the following instruction to retrieve it 
# qq = gl.SFrame(“SP500_daily.bin/”)

# import matplotlib.pyplot as plt
# %matplotlib inline # only for those who are using IPython notebook
# plt.plot(qq['close'])

# #------------------------------------------------
# # Adding Outcome
# #------------------------------------------------

# # add the outcome variable, 1 if the trading session was positive (close>open), 0 otherwise
# qq['outcome'] = qq.apply(lambda x: 1 if x['close'] > x['open'] else -1)
# # we also need to add three new columns ‘ho’ ‘lo’ and ‘gain’
# # they will be useful to backtest the model, later
# qq['ho'] = qq['high'] - qq['open'] # distance between Highest and Opening price
# qq['lo'] = qq['low'] - qq['open'] # distance between Lowest and Opening price
# qq['gain'] = qq['close'] - qq['open']

# ts = gl.TimeSeries(qq, index='datetime')
# # add the outcome variable, 1 if the bar was positive (close>open), 0 otherwise
# ts['outcome'] = ts.apply(lambda x: 1 if x['close'] > x['open'] else -1)

# # GENERATE SOME LAGGED TIMESERIES
# ts_1 = ts.shift(1) # by 1 day
# ts_2 = ts.shift(2) # by 2 days
# # ...etc....
# # it's an arbitrary decision how many days of lag are needed to create a good forecaster, so
# # everyone can experiment by his own decision

# #------------------------------------------------
# # Adding Predictors
# #------------------------------------------------


# ts['feat1'] = ts['close'] > ts_1['close']
# ts['feat2'] = ts['close'] > ts_2['close']

# # add_features is a helper function, which is out of the scope of this article,
# # and it returns a tuple with:
# # ts: a timeseries object with, in addition to the already included columns, also lagged columns
# # as well as some features added to train the model, as shown above with feat1 and feat2 examples
# # l_features: a list with all features used to train Classifier models
# # l_lr_features: a list all features used to train Linear Regression models

# ts, l_features, l_lr_features = add_features(ts)

# # add the gain column, for trading operations with LONG only positions. 
# # The gain is the difference between Closing price - Opening price
# ts['gain'] = ts['close'] - ts['open']

# ratio = 0.8 # 80% of training set and 20% of testing set
# training = ts.to_sframe()[0:round(len(ts)*ratio)]
# testing = ts.to_sframe()[round(len(ts)*ratio):]

# #------------------------------------------------
# # Training a Decision Tree Model
# #------------------------------------------------

# max_tree_depth = 6
# decision_tree = gl.decision_tree_classifier.create(training, validation_set=None, 
#                                                    target='outcome', features=l_features, 
#                                                    max_depth=max_tree_depth, verbose=False)

# #------------------------------------------------
# # Measuring Performance of Fitted Model
# #------------------------------------------------


# decision_tree.evaluate(training)['accuracy'], decision_tree.evaluate(testing)['accuracy']

# #------------------------------------------------
# # Predicting Data
# #------------------------------------------------


# predictions = decision_tree.predict(testing)
# # and we add the predictions  column in testing set
# testing['predictions'] = predictions

# # let's see the first 10 predictions, compared to real values (outcome column)
# testing[['datetime', 'outcome', 'predictions']].head(10)

# #------------------------------------------------
# # Backtesting the Model
# #------------------------------------------------


# pnl = testing[testing['predictions'] == 1]['gain'] # the gain column contains (Close - Open) values
# # I have written a simple helper function to plot the result of all the trades applied to the
# # testing set and represent the total return expressed by the index basis points
# # (not expressed in dollars $)
# plot_equity_chart(pnl,'Decision tree model')


# #------------------------------------------------
# # Trading Basics
# #------------------------------------------------


# # This is a helper function to trade 1 bar (for example 1 day) with a Buy order at opening session
# # and a Sell order at closing session. To protect against adverse movements of the price, a STOP order
# # will limit the loss to the stop level (stop parameter must be a negative number)
# # each bar must contains the following attributes: 
# # Open, High, Low, Close prices as well as gain = Close - Open and lo = Low - Open
# def trade_with_stop(bar, slippage = 0, stop=None):
#     """
#     Given a bar, with a gain obtained by the closing price - opening price
#     it applies a stop limit order to limit a negative loss
#     If stop is equal to None, then it returns bar['gain']
#     """
#     bar['gain'] = bar['gain'] - slippage
#     if stop<>None:
#         real_stop = stop - slippage
#         if bar['lo']<=stop:
#             return real_stop
#     # stop == None    
#     return bar['gain']

# #------------------------------------------------
# # Trading Costs
# #------------------------------------------------

# SLIPPAGE = 0.6
# STOP = -3
# trades = testing[testing['predictions'] == 1][('datetime', 'gain', 'ho', 'lo', 'open', 'close')]
# trades['pnl'] = trades.apply(lambda x: trade_with_stop(x, slippage=SLIPPAGE, stop=STOP))
# plot_equity_chart(trades['pnl'],'Decision tree model')
# print("Slippage is %s, STOP level at %s" % (SLIPPAGE, STOP))

# predictions_prob = decision_tree.predict(testing, output_type = 'probability')
# # predictions_prob will contain probabilities instead of the predicted class (-1 or +1)

# trades = testing[predictions_prob>=0.5][('datetime', 'gain', 'ho', 'lo', 'open', 'close')]

# Net gain = (Gross gain - SLIPPAGE) * MULT - 2 * COMMISSION

# drawdown = pnl - pnl.cumulative_max()
# max_drawdown = min(drawdown)

# model = decision_tree
# predictions_prob = model.predict(testing, output_type="probability")
# THRESHOLD = 0.5
# bt_1_1 = backtest_ml_model(testing, predictions_prob, target='outcome',  
#                            threshold=THRESHOLD, STOP=-3, 
#                            MULT=25, SLIPPAGE=0.6, COMMISSION=1, plot_title='DecisionTree')
# backtest_summary(bt_1_1)


# THRESHOLD = 0.55 
# # it’s the minimum threshold to predict an Up day so hopefully a good day to trade
# bt_1_2 = backtest_ml_model(testing, predictions_prob, target='outcome',  
#                            threshold=THRESHOLD, STOP=-3, 
#                            MULT=25, SLIPPAGE=0.6, COMMISSION=1, plot_title='DecisionTree')
# backtest_summary(bt_1_2)

# #------------------------------------------------
# # Training a Logistic Classifier
# #------------------------------------------------


# model = gl.logistic_classifier.create(training, target='outcome', features=l_features, 
#                                       validation_set=None, verbose=False)
# predictions_prob = model.predict(testing, 'probability')
# THRESHOLD = 0.6
# bt_2_2 = backtest_ml_model(testing, predictions_prob, target='outcome', 
#                            threshold=THRESHOLD, STOP=-3, plot_title=model.name())
# backtest_summary(bt_2_2)

# #------------------------------------------------
# # Training a Linear Regression Model
# #------------------------------------------------

# model = gl.linear_regression.create(training, target='gain', features = l_lr_features,
#                                    validation_set=None, verbose=False, max_iterations=100)
# predictions = model.predict(testing)
# # a linear regression model, predict continuous values, so we need to make an estimation of their
# # probabilities of success and normalize all values in order to have a vector of probabilities
# predictions_max, predictions_min = max(predictions), min(predictions)
# predictions_prob = (predictions - predictions_min)/(predictions_max - predictions_min)


# THRESHOLD = 0.4
# bt_3_2 = backtest_linear_model(testing, predictions_prob, target='gain', threshold=THRESHOLD,
#                               STOP = -3, plot_title=model.name())
# backtest_summary(bt_3_2)

# #------------------------------------------------
# # Training a Boosted Tree
# #------------------------------------------------

# model = gl.boosted_trees_classifier.create(training, target='outcome', features=l_features, 
#                                            validation_set=None, max_iterations=12, verbose=False)
# predictions_prob = model.predict(testing, 'probability')

# THRESHOLD = 0.7
# bt_4_2 = backtest_ml_model(testing, predictions_prob, target='outcome', 
#                            threshold=THRESHOLD, STOP=-3, plot_title=model.name())
# backtest_summary(bt_4_2)

# #------------------------------------------------
# # Training a Random Forest
# #------------------------------------------------


# model = gl.random_forest_classifier.create(training, target='outcome', features=l_features, 
#                                       validation_set=None, verbose=False, num_trees = 10)
# predictions_prob = model.predict(testing, 'probability')
# THRESHOLD = 0.6
# bt_5_2 = backtest_ml_model(testing, predictions_prob, target='outcome', 
#                            threshold=THRESHOLD, STOP=-3, plot_title=model.name())
# backtest_summary(bt_5_2)