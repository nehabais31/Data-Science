"""
Neha Bais
MET CS-677 Assignment-7.1

Implementing Linear Regression classifier

1. Using w[5 to 30], calculate P/L per trade using tradig strategy and find optimal w.
2. Use optimal w from 2018 and compute r**2. 
3. Use optimal w from 2018 and implement trading for 2019. Find how many long 
   and short positions we have for 2019.
4. Compute average profit/loss per "long position" trade and per "short position" 
   trades in 2019?
5. Find avg number of days for long position and short position transactions in 2019.
6. Compare results of 2018 & 2019 using optimal w.

"""

import numpy as np
import os
import pandas as pd
import matplotlib . pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
from collections import Counter


ticker = 'BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
ticker_file = os.path.join(input_dir, ticker + '.csv')

 
def calculate_loss_function(month_data) :
    
    X = np.arange(1,len(month_data)+1).reshape(-1,1)
    Y = month_data
    model = LinearRegression(fit_intercept=True)
    model.fit(X, Y)
    
    for i in range(len(Y)):
        squared_errors = (Y - model.predict(X)) ** 2
    return round( np.sum(squared_errors) , 2)  


def break_day_candidate(month_data, break_day, sse_t1, sse_t2 ) :
    L1 = {}     # for calculating loss function for 1st regression
    L2 = {}     # for calculating loss function for 2nd regression
                       
    # splitting data into 2 parts to calculate 2 regression lines
    for k in range(2, len(month_data)-1) :
                
        m_1 = month_data[ : k]
        m_2 = month_data[k : ]
            
        L1[k] = calculate_loss_function(m_1)
        L2[k] = calculate_loss_function(m_2)
            
            
        total_loss_2018 = dict(Counter(L1) + Counter(L2))
        break_day[m] = [k for k, v in total_loss_2018.items() 
                         if v == min(total_loss_2018.values())][0]
        
        sse_t1[m] = L1[break_day[m]]
        sse_t2[m] = L2[break_day[m]]
        
    return  break_day,  sse_t1, sse_t2
        
    

try :
    df = pd.read_csv(ticker_file)
        
    # extract 2018 and 2019 data separately
    data_2018 = df.loc[df['Year'] == 2018].copy()
    data_2019 = df.loc[df['Year'] == 2019].copy()
    
    sse_2018 = {}
    sse_t1_18 = {}
    sse_t2_18 = {}
    
    k_2018 = {}
    sse_2019 = {}
            
    
    for m in range(1,13):
        month_data_2018 = data_2018[data_2018['Month'] == m]['Adj Close'].values
        month_data_2019 = data_2019[data_2019['Month'] == m]['Adj Close'].values
                
        sse_2018[m] = calculate_loss_function(month_data_2018)
        sse_2019[m] = calculate_loss_function(month_data_2019)
        
        k_2018, sse_t1_18, sse_t2_18 = break_day_candidate(month_data_2018, k_2018,
                                                         sse_t1_18, sse_t2_18 )
        
        
        
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)        