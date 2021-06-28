"""
Created on Tue Jun  2 11:04:44 2020

@author: Neha Bais
MET CS-677 Assignment-2.3 
Last Digit Analysis

1. what is the most frequent digit?

2. what is the least frequent digit?

3. compute the following 4 error metrics for your data:
(a) max absolute error
(b) median absolute error
(c) mean absolute error
(d) root mean squared error (RMSE)

"""

import pandas as pd
import numpy as np
import os


ticker = 'BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
ticker_file = os.path.join(input_dir, ticker + '.csv')

def max_abs_error(actual, predicted) :
    # returns max absolute error 
    # max(|a1 - p1|,.........|an - pn|)
    
    max_abs_err = np.max(abs(actual - predicted))
    return round(max_abs_err,2)

def median_abs_error(actual, predicted) :
    # returns median absolute error
    # median(|a1 - p1|,.........|an - pn|)
    
    median_abs_err = np.median(abs(actual - predicted))
    return round(median_abs_err,2)

def mean_abs_error(actual, predicted) :
    # returns mean absolute error
    # ((|a1 - p1|,.........|an - pn|) / n)
    
    mean_abs_err = np.mean(abs(actual - predicted))
    return round(mean_abs_err,2)
    
def rmse(actual, predicted) :
    # returns mean square root error
    
    rms_err = np.sqrt(np.mean((actual - predicted)**2))
    return round(rms_err,2)


try :
    df = pd.read_csv(ticker_file)
    
    df['Open'] = df['Open'].round(2)
    
    # Extract last digit from Open Price
    df['last_digit'] = df['Open'].apply(lambda x: str(x)[-1]).astype(int)
        
    df['Count'] = 1
    df['Year Count'] = 1
        
    # Calculating total number of digits group by each digit in the last digit column
    total_digit = df.groupby('last_digit')['Count'].sum()
    #print(total_digit)
    
    # Actual last digit vector %
    actual = (total_digit/len(df)) * 100
          
    # Prediction vector
    # Assuming that each digit is equally likely and occurs 10% of the time
    predicted_last_digit = np.array(10 * [10])
    
    # Ques-1
    print('\nQuestion-1')
    print('Most frequent last digit is: ', total_digit.argmax(), 'occuring ', max(total_digit), 'times.')
    
    # Ques-2
    print('\nQuestion-2')
    print('Least frequent last digit is: ', total_digit.argmin(), 'occuring ', min(total_digit), 'times.')
    
    # Ques-3
    # Calculatinf\g total number of digits for each year group by each digit in the last digit column
    total_digit_yearly = df.groupby(['Year','last_digit'])['Count'].sum()
    #print(total_digit_yearly)
    
    # Actual last digit yearly % 
    # year_count = nbr of days in a year
    year_count = df.groupby('Year')['Year Count'].count()
    
    actual_yearly = ( total_digit_yearly / year_count) * 100
    #print(actual_yearly)
    
    year_list = df['Year'].unique()
    
    error_metric = {}   # dict to store error metrics for each year
    
    
    # Calculating error metrics for each year and storing results in a dict
    for year in year_list :
        error_metric[year] = [max_abs_error(actual_yearly[year], predicted_last_digit), 
                              median_abs_error(actual_yearly[year], predicted_last_digit),
                              mean_abs_error(actual_yearly[year], predicted_last_digit),
                              rmse(actual_yearly[year], predicted_last_digit)]
     
    
        
    # error metric data frame
    error_metric_df = pd.DataFrame.from_dict(error_metric, orient = 'index',
                                             columns = [ 'max abs error', 'median abs error', 'mean abs error' , 'RMSE' ])
    
    # printing dataframe object on console
    print('\nQuestion-3')
    print(error_metric_df)
    print('\nConclusion: \nAs per the mteric error values, the actual values for last digit'
          ' does not vary at a greater extent from the predicted values.'
          ' In this case, the RMSE is around 3 ~ 4%, so we can say that our prediction is not 100% correct but it is also not deviating at a larger extent.')
    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)    
