"""
Neha Bais
MET CS-677 Assignment-7.2

Change of trend detection

1. For 2018 & 2019, fid the breakdown day and decide whtether there is 
   a pricing trend change for each month.
2. How many months exhibit trend change
3. Compare results of both the years.   

"""

import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from collections import Counter
from scipy.stats import f as fisher_f


ticker = 'BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
ticker_file = os.path.join(input_dir, ticker + '.csv')

 
def calculate_loss_function(month_data) :
    '''
    Returns the loss function
    '''
    
    X = np.arange(1,len(month_data)+1).reshape(-1,1)
    Y = month_data
    model = LinearRegression(fit_intercept=True)
    model.fit(X, Y)
    
    for i in range(len(Y)):
        squared_errors = (Y - model.predict(X)) ** 2
    return round( np.sum(squared_errors) , 2)  


def break_day_candidate(month_data, break_day, sse_t1, sse_t2 ) :
    '''
    Computing the breakdown day giving the minimum
    loss function for 2 regressions
    '''
    
    L1 = {}     # for calculating loss function for 1st regression
    L2 = {}     # for calculating loss function for 2nd regression
                       
    # splitting data into 2 parts to calculate 2 regression lines
    for k in range(2, len(month_data)-1) :
                
        m_1 = month_data[ : k]
        m_2 = month_data[k : ]
            
        L1[k] = calculate_loss_function(m_1)
        L2[k] = calculate_loss_function(m_2)
            
        # total loss for both the regressions    
        total_loss_2018 = dict(Counter(L1) + Counter(L2))
        
        # finding breakdown day with minimum value of total loss for each month
        break_day[m] = [k for k, v in total_loss_2018.items() 
                         if v == min(total_loss_2018.values())][0]
        
        # Loss function corresponding to breakdown day
        sse_t1[m] = L1[break_day[m]]
        sse_t2[m] = L2[break_day[m]]
               
    return  break_day,  sse_t1, sse_t2
        
    

try :
    df = pd.read_csv(ticker_file)
        
    # extract 2018 and 2019 data separately
    data_2018 = df.loc[df['Year'] == 2018].copy()
    data_2019 = df.loc[df['Year'] == 2019].copy()
    
    # Dictionaries to hold different values for each month of both the years
    sse_2018 = sse_2019 = {}
    sse_t1_18 = sse_t1_19 = {}
    sse_t2_18 = sse_t2_19 = {}
    
    k_2018 = k_2019 = {}
    f_stats_18 = f_stats_19 = {}
    p_value_18 = p_value_19 = {}
                
    
    for m in range(1,13):
        month_data_2018 = data_2018[data_2018['Month'] == m]['Adj Close'].values
        month_data_2019 = data_2019[data_2019['Month'] == m]['Adj Close'].values
         
        # Calculating the loss function for each month for single regression
        sse_2018[m] = calculate_loss_function(month_data_2018)
        sse_2019[m] = calculate_loss_function(month_data_2019)
        
        '''
        Computing the breakdown day using 2 regressions
        Also loss function corresponding to the breakdown day 
        for both regressions
        '''
        k_2018, sse_t1_18, sse_t2_18 = break_day_candidate(month_data_2018, k_2018,
                                                         sse_t1_18, sse_t2_18 )
        
        k_2019, sse_t1_19, sse_t2_19 = break_day_candidate(month_data_2019, k_2019,
                                                         sse_t1_19, sse_t2_19 )
        
        ##################################
        #    2018 - fstats and p value   #
        ##################################
        f_stats_18[m] = ((sse_2018[m] - (sse_t1_18[m] + sse_t2_18[m])) / 2) * \
            ((sse_t1_18[m] + sse_t2_18[m]) / (len(month_data_2018) - 4))**(-1)
            
        p_value_18[m] = fisher_f.cdf(f_stats_18[m], 2,len(month_data_2018)-4 ) 
        
        
        ##################################
        #    2019 - fstats and p value   #
        ##################################
        f_stats_19[m] = ((sse_2019[m] - (sse_t1_19[m] + sse_t2_19[m])) / 2) * \
            ((sse_t1_19[m] + sse_t2_19[m]) / (len(month_data_2019) - 4))**(-1)
            
        p_value_19[m] = fisher_f.cdf(f_stats_19[m], 2,len(month_data_2019)-4 )
        
        
    #######################
    #     Conclusions     #
    #######################
    res_2018 = ['2 regressions better' if v > 0.1 else '1 regression is better' \
                for k,v in p_value_18.items()]
            
    res_2019 = ['2 regressions better' if v > 0.1 else '1 regression is better' \
                for k,v in p_value_19.items()]
    
    p_data = {
             '2018': res_2018,
             '2019':  res_2019 }
    
    p_df = pd.DataFrame(p_data,columns = ['2018','2019'],
                        index = ['Jan','Feb','Mar','Apr','May','Jun',
                                 'Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    print('\t\tTrend change results')
    print('\t\t--------------------')
    print(p_df)
     
    print('\nConclusion')
    print('1. From the above results it is clear that for both the years, there '
          'is a change in trend of pricing. \n'
          '2. Trend change is observed for all the months. \n'
          '3. Results are same for both the years.')
       
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)        