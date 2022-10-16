import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score

def ForestRegressor(start,end) :
    #Imports the data
    data = pd.read_csv (r'C:\Users\meht1\OneDrive\Desktop\MATH3599\Index_Analysis\IndexFutures\IndexFutures\ContractFutures_1min_uic4039.csv')
    
    #Creates a copy so we do not modify the original
    #df = data
    df2 = data.copy()
    df2 = df2.iloc[start:end,:]
    
    #Obtains descriptive statisitcs
    #print(df2.describe()) 
    
    #Pulls out the time column and splits the time into seperate components (Used if implemented into PowerBi)
    #date_col = pd.to_datetime(df["Time"])
    #del df["Time"]
    #df["Year"] = date_col.dt.year
    #df["Month"] = date_col.dt.month
    #df["Day"] = date_col.dt.day
    #df["Time"] = date_col.dt.time
    
    #Creates a line chart of the closing sale data
    #df.plot(y = "Close", use_index=True);
    
    # Modifying the data
    df2.rename( columns={'Unnamed: 0':'Count'}, inplace=True )
    y = df2["Close"]
    del df2["Close"]
    del df2["Time"]
    
    #Splitting the data set
    X_train, X_test, y_train, y_test = train_test_split(df2,y,test_size = 0.20)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    #Instantation of the model
    model = RandomForestRegressor(n_estimators = 300, random_state= 30)
    
    #Fitting the model
    fit_close = model.fit(X_train, y_train)
    
    print("Training Model Evaluation")
    #Training model evaluation
    print('The Training of r_sq is: %.2f'% fit_close.score(X_train, y_train)) #R^2 value
    yTrain_pred = fit_close.predict(X_train)
    #MAE 
    print("The MAE is %.2f"% mean_absolute_error(y_train, yTrain_pred))
    #MSE
    print("The MSE is %.2f"% mean_squared_error(y_train, yTrain_pred))
    #RMSE 
    print("The RMSE is %.2f"% np.sqrt(mean_squared_error(y_train, yTrain_pred)))
    #EVS
    print("The EVS is %.2f"% explained_variance_score(y_train, yTrain_pred))
    
    print("Testing Data Evaluation")
    #Predicting the on testing data
    yTest_pred = fit_close.predict(X_test)
    #Co-efficient of determinitation
    print("The co-efficient of determination is %.2f"% r2_score(y_test, yTest_pred))
    #MAE
    print("The MAE is %.2f"% mean_absolute_error(y_test, yTest_pred))
    #MSE
    print("The MSE is %.2f"% mean_squared_error(y_test, yTest_pred))
    #RMSE 
    print("The RMSE is %.2f"% np.sqrt(mean_squared_error(y_test, yTest_pred)))
    #EVS
    print("The EVS is %.2f"% explained_variance_score(y_test, yTest_pred))
    
    #Plotting the observed and predicted data
   # plt.rcParams['figure.figsize'] = (10,6)
    x_ax = range(len(X_test))
    #Plotting
    plt.plot(x_ax, y_test, label = 'Observed', color = 'black', linestyle = '-')
    plt.plot(x_ax, yTest_pred, label = 'Predicted', color = 'red', linestyle = '--')
    
ForestRegressor(50,200)


