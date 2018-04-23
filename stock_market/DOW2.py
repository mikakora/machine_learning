 # DOW1 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from pandas_talib import *
#import talib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import log_loss

dow = pd.read_csv('C:/ML2018/data/DOW_AC.csv')  # Adjusted Close
dow_O = pd.read_csv('C:/ML2018/data/DOW_O.csv') # Open
dow_L = pd.read_csv('C:/ML2018/data/DOW_L.csv') # Low
dow_H = pd.read_csv('C:/ML2018/data/DOW_H.csv') # High
dow_C = pd.read_csv('C:/ML2018/data/DOW_C.csv') # Close
dow_V = pd.read_csv('C:/ML2018/data/DOW_V.csv') # Volume

tickers = list(dow.columns[1:]) 
# calculate one-day returns
ret1 = dow[tickers] / dow[tickers].shift(1) - 1

# compute array of different lookback returns
lb = [2,3,5,8,13,21,34,55,89,144,233]  # Fibonacci 
#lb = list(range(2,20)) + list(np.linspace(21,252,12))  # Takeuchi article

rets=ret1
for k in lb:
    rets=np.append(rets,dow[tickers] / dow[tickers].shift(k) - 1,axis=1)   
    
rets = rets[253:]  # remove 1st year of data

nd = rets.shape[0]   # number of days
nt = len(tickers)    # number of tickers 
for i in range(nd):
    fa = np.reshape(rets[i],(-1,nt)).transpose()
    rets[i] = np.reshape(fa,(-1,))    
  
X0 = np.reshape(rets,(-1,len(lb)+1))
"""
# compute labels for look ahead (la) of 1 day 
la = 1;  # predict one-day ahead
ns = la*nt  # number of columns to skip
X = X0[0:-ns]   # X is the features data set
y = X0[ns:,0] > 0   # y is the labels data set

# compute labels for look ahead (la) of 1 week (5-day) 
la = 5;  # predict 5-day ahead
ns = la*nt  # number of columns to skip
X = X0[0:-ns]   # X is the features data set
y = X0[ns:,3] > 0   # y is the labels data set
"""
# compute labels for look ahead (la) of 1 month (21-day)
la = 21;  # predict 21-day ahead
ns = la*nt  # number of columns to skip
X = X0[0:-ns]   # X is the features data set
y = X0[ns:,6] > 0  # y is the labels data set

train_size = 180000  # 5000 days x 36 tickers
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

# Random Forest
rf_model0 = RandomForestClassifier(n_estimators = 100,n_jobs=-1,warm_start=True,random_state=7,verbose=1) # define model
errors = []
for i in range(10):
    rf_model0.fit(X_train, y_train)
    rf_model0.n_estimators += 100
    errors.append(log_loss(y_test, rf_model0.predict_proba(X_test)))

_ = plt.plot(errors, '-r')

acc1 = rf_model0.score(X_train, y_train)
acc2 = rf_model0.score(X_test, y_test)

# Gradient Boosting Classifier
gb_model0 = GradientBoostingClassifier(n_estimators = 10,warm_start=True,random_state=7,verbose=1)
errors = []
for i in range(10):
    gb_model0.fit(X_train, y_train)
    gb_model0.n_estimators += 10
    errors.append(log_loss(y_test, gb_model0.predict_proba(X_test)))

_ = plt.plot(errors, '-r')

acc1 = gb_model0.score(X_train, y_train)
acc2 = gb_model0.score(X_test, y_test)

# LinearDiscriminantAnalysis
lda_model0 = LinearDiscriminantAnalysis()
lda_model0.fit(X_train, y_train)
loss = log_loss(y_test, lda_model0.predict_proba(X_test))

loss

acc1 = lda_model0.score(X_train, y_train)
acc2 = lda_model0.score(X_test, y_test)

# QuadraticDiscriminantAnalysis
qda_model0 = QuadraticDiscriminantAnalysis()
qda_model0.fit(X_train, y_train)
loss = log_loss(y_test, qda_model0.predict_proba(X_test))

loss
acc1 = qda_model0.score(X_train, y_train)
acc2 = qda_model0.score(X_test, y_test)

