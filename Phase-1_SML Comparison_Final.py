import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC

#------------------------------------------------------------------------------
'''
Data Specifications:
    
    UIC	     Currency	            Description
    9046 	   AUD	       ASX SPI 200 Index - continuous
    4044       GBP	         FTSE 100 Index - continuous
    4039  	   USD	     E-mini S&P 500 (Dollar) - continuous
    
NOTE:    
      1.    Time Stamps are standardised to same time code
''' 
#------------------------------------------------------------------------------

# Specify input File ( "ASX", "FTSE" , "S&P")
input = "ASX"

#Input Number of Rows (10% of data ( 6 months ) approx. 100,000 rows)
nr = 5000

# Start Analysis after N rows (1 Trading year approx 250K rows)
skip = 0

# Test vs. Train Split  - 0.2 represents 20% Train - 80% Test
test = 0.8

#------------------------------------------------------------------------------
#Select input file - NOTE: New user must respecify input location for each file

if(input == "ASX"):
    data=pd.read_csv(r'C:\Users\lenno\OneDrive\Desktop\IndexFutures\ContractFutures_1min_uic9046.csv', nrows=nr, skiprows=skip)
    input = "ASX SPI 200 Index"
    nrtotal=1583999
elif (input =="FTSE"):
    data=pd.read_csv(r'C:\Users\lenno\OneDrive\Desktop\IndexFutures\ContractFutures_1min_uic4044.csv', nrows=nr, skiprows=skip)
    input = "FTSE 100 Index"
    nrtotal=1451999
elif (input =="S&P") :
    data = pd.read_csv(r'C:\Users\lenno\OneDrive\Desktop\IndexFutures\ContractFutures_1min_uic4039.csv', nrows=nr, skiprows=skip)
    input= "E-mini S&P 500 (Dollar)"
    nrtotal=1583999
    
#Name columns
col_names = ['Item','Close','High','Interest','Low','Open','Time','Volume']

#Analysis Start & End Date
first, last = data.Time[~data.Time.isna()].values[[0, -1]]
data.columns = col_names
data.set_index('Item')

#select relevant columns for analysis
data = data.drop(columns = ['High','Interest','Low','Open','Time'])
percentdataused = nr/1048576

#Print Model Input Settings
print("\nInput File: \n\n",input,"\n")
print("Analysis Start Date:\n\n",first,"\n")
print("Analysis End Date:\n\n",last,"\n")
print("Percentage of Data Used: \n \n ","{:.0%}".format(percentdataused),"\n")

#Define model parameters
def model_variables(prices,lags):
    
    '''
    Parameters:
        prices: dataframe with 6 years of Historical index equity future market data
        with the variables close price, low, high, Interest and volume.
        lags: 2 lag variables are computed using the closing price to create lagged 
        returns variables or features a given model
    Output:
        tsret: dataframe with numerical index value for the independent variables(X) and the 
        dependent variable(y) for a given model.
    '''
    
    # Change data types of prices dataframe from object to numeric
    prices = prices.apply(pd.to_numeric)
    
    # Create the new lagged DataFrame
    inputs = pd.DataFrame(index=prices.index)
    inputs["Close"] = prices["Close"]
    inputs["Volume"] = prices["Volume"]
    
    # Create the shifted lag series of prior trading period close values
    for i in range(0, lags):
        
        tsret = pd.DataFrame(index=inputs.index)        
        inputs["Lag%s" % str(i+1)] = prices["Close"].shift(i+1)
   
    #Create the returns DataFrame   
    tsret["VolumeChange"] =inputs["Volume"].pct_change()    
    tsret["returns"] = inputs["Close"].pct_change()*100.0
        
    # If any of the values of percentage returns equal zero, set them to
    # approximatley zero to prevent issues with sklearn
    
    for i,x in enumerate(tsret["returns"]):
        
        if (abs(x) < 0.0001):
            
            tsret["returns"][i] = 0.0001
    
    # Create the lagged percentage returns columns
    
    for i in range(0, lags):
        
        tsret["Lag%s" % str(i+1)] = \
          inputs["Lag%s" % str(i+1)].pct_change()*100.0
    
    # Create the "Direction" column (+1 or -1) indicating an up/down day
    # drop any  cases where price direction is equal to zero
    tsret = tsret.dropna()    
    tsret["Direction"] = np.sign(tsret["returns"])
    
    # Convert index to datetime in order to filter the dataframe by dates when 
    # we create the train and test dataset
        
    return tsret

# Pass the dataset(data) and the number of lags 2 as the inputs of the model_variables  function

variables_data = model_variables(data,2)
 
# Use the prior two days of returns and the volume change as predictors
# values, with direction as the response

dataset = variables_data[["Lag1","Lag2","VolumeChange","Direction"]]


dataset = dataset.dropna()
dataset = dataset[np.isfinite(dataset).all(1)]

# Create independent variable X and depedendent variable Y
X = dataset[["Lag1","Lag2","VolumeChange"]]
y = dataset["Direction"]

#Print Output 
print("Training vs. Testing Split Selected: \n \n", "{:.0%}".format(test),"vs.","{:.0%}".format(1-test),"\n")

#split data set into test and train
data_split = int(nr*test)
X_train = X[X.index <= data_split]
X_test =  X[X.index > data_split]
y_train = y[y.index <= data_split]
y_test = y[y.index > data_split]
 
# Create the (parametrised) models
print("Hit Rates/Confusion Matrices:\n")
models = [("LR", LogisticRegression(),"Logistic Regression"),
              ("LDA", LDA(),"Linear Discriminant Analysis"),
              ("SVM", SVC(
                      C=1000000.0, cache_size=200, class_weight=None,
                      coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
                      max_iter=-1, probability=False, random_state=None,
                      shrinking=True, tol=0.001, verbose=False)
    ,"Support Vector Classifier"),
    ("RF", RandomForestClassifier(
            n_estimators=100, criterion='gini',
            max_depth=None, min_samples_split=2,
            min_samples_leaf=30, max_features='auto',
            bootstrap=True, oob_score=False, n_jobs=1,
            random_state=None, verbose=0),"Random Forest Classifier"
    )]
 
# Use Loop for each model to compute and visualise the results
for m in models:
    
    # Train each model using training set, create Array of Test Set Predictions
    m[1].fit(X_train, y_train)
    pred = m[1].predict(X_test)
    
    # Print Results 
    print("%s:" % m[0])
    print('Recall: %.3f' % recall_score(pred, y_test))	
    print('Accuracy: %.3f' % accuracy_score(pred, y_test))  
    print('F1 Score: %.3f' % f1_score(pred, y_test))
    print("%s\n" % confusion_matrix(pred, y_test))
    
    # Results Visulation
    cm = confusion_matrix(pred, y_test) 
    group_names = ['True Negative','False Positive','False Negative', 'True Positive']
    group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                      cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    name = ['Decrease', 'Increase']
    ax= plt.subplot()
    fig = sns.heatmap(cm, fmt='', annot=labels, cmap='Blues',ax=ax)
    ax.set_xlabel('Predicted Direction');
    #Manually change index name output in Visualation
    ax.set_title("ASX - " + m[2]); 
    ax.set_ylabel('Actual Direction'); 
    ax.xaxis.set_ticklabels(['Down', 'Up']); 
    ax.yaxis.set_ticklabels(['Down', 'Up']);
    #Manually change index name output file name for Visualation
    fig.figure.savefig(("test _{m[0]}.png".format(m=m)))
    fig.figure.clf()
       
