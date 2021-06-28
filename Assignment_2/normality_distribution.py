"""
Created on Mon Jun  1 11:15:58 2020

Neha Bais
CS677 Assignment 2.2 - Normality Distribution for BAC stock

1. Compute nbr of days with +ve and -ve returns

2. Compute % of days with returns > mean and returns < mean 

3.Compute nbr of days when abs(returns) > [mean + 2(std)] and abs(returns) < [mean - 2(std)]

"""


import pandas as pd
import numpy as np
import os


ticker = 'BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
ticker_file = os.path.join(input_dir, ticker + '.csv')

try :
    df = pd.read_csv(ticker_file)
    df['Return'] = 100 * df['Return']
    
    # Calculting average returns for each year
    avg_returns = df.groupby(['Year'])['Return'].mean()
    
    # Calculating total number of trading days per year
    trading_days = df.groupby(['Year'])['Return'].count()
    
    # unique years in our data
    year_list = df['Year'].unique()
        
    days_above_mean = []
    days_below_mean = []
    
    days_gt_zero = []
    days_lt_zero = []
    
    # Checking days whose return is greter or smaller than avergae returns year wise
    for year in year_list:
        # Storing days above mean in a list
        above_mean = df.loc[df['Year'] == year]['Return'] > avg_returns[year]
        days_above_mean.extend(above_mean)
             
        # Storing days below mean in a separate list
        below_mean = df.loc[df['Year'] == year]['Return'] < avg_returns[year]
        days_below_mean.extend(below_mean)
             
        # days when return > 0
        pos_days = df.loc[df['Year'] == year]['Return'] > 0
        days_gt_zero.extend(pos_days)
                
        # days when return < 0
        neg_days = df.loc[df['Year'] == year]['Return'] < 0
        days_lt_zero.extend(neg_days)
             
          
    # Adding days above and below mean to data frame    
    df['Above Mean'] = days_above_mean
    df['Below Mean'] = days_below_mean
       
    df['Days > 0'] = days_gt_zero
    df['Days < 0'] = days_lt_zero
        
    # Get the count of nbr of days above and below mean year wise        
    nbr_of_days_above_mean = df.groupby('Year')['Above Mean'].sum().astype(int)
    nbr_of_days_below_mean = df.groupby('Year')['Below Mean'].sum().astype(int)
    
        # Number of days > 0 year wise
    nbr_of_days_gt_zero = df.groupby('Year')['Days > 0'].sum().astype(int)
    #print(nbr_of_days_gt_zero)
    
    # Number of days < 0 year wise
    nbr_of_days_lt_zero = df.groupby('Year')['Days < 0'].sum().astype(int)
    #print(nbr_of_days_lt_zero)
    
    
    # Calculating % days above and below mean
    pct_days_above_mean = round((nbr_of_days_above_mean / trading_days) * 100 , 2)
    pct_days_below_mean = round((nbr_of_days_below_mean / trading_days) * 100 , 2)
    
    
    # Creating dataframe for Ques-1
    df_1 = pd.DataFrame(columns = ['Year' , 'Trading days' , 'Positive return Days', 'Negative return Days'])
    df_1['Year'] = year_list
    df_1['Trading days'] = list(trading_days)
    df_1['Positive return Days'] = list(nbr_of_days_gt_zero)    
    df_1['Negative return Days'] = list(nbr_of_days_lt_zero)
    
    
    # Creating df for values required for Ques-2
    df_2 = pd.DataFrame(columns = ['Year', 'Trading days', 'μ', '% days < μ', '% days > μ'])
    df_2['Year'] = year_list
    df_2['Trading days'] = list(trading_days)
    df_2['μ'] = list(avg_returns)
    df_2['% days < μ'] = list(pct_days_below_mean)
    df_2['% days > μ'] = list(pct_days_above_mean)
    
        
    # Printing output to console - Ques-1 & Ques-2
    #------------ Ques: 1-----------------
    print('\nQuestion-1')
    print('**********')
    pos_return = len(df[df['Return'] > 0])
    neg_return = len(df[df['Return'] < 0])
    print('Total days with +ve returns: ', pos_return)
    print('Total days with -ve returns: ', neg_return)
    print()
    print(df_1.to_string(index = False))
    
    #------------ Ques: 2------------------
    print('\nQuestion-2')
    print('**********')
    print(df_2.to_string(index = False))
    
    print('\n*********** Conclusion *******************')
    print('1. In my case', '+ve days are greater than -ve days.' if pos_return > neg_return 
          else '-ve days are greater than +ve days.')
    print('2. These +ve and -ve days are not consistent per year. For some years positive days are greater'
          ' while in other years, negative days are greater.')
          
       
    #---------------- Ques: 3------------------------
    
    # Calculating std deviations returns for each year
    std_returns = df.groupby('Year')['Return'].std()
        
    # Calculate μ+2σ / μ-2σ
    avg_plus_2std  = avg_returns + (2 * std_returns)
    avg_minus_2std = avg_returns - (2 * std_returns) 
    
    # Temporary lists to store nbr of days where returns > μ+2σ and returns < μ-2σ
    days_above_2std = []
    days_below_2std = []
    
    # Checking days per year where returns > μ+2σ and returns < μ-2σ    
    for year in year_list:
        # Storing days above μ+2σ in a list
        above_2std = df.loc[df['Year'] == year]['Return'] > avg_plus_2std[year]
        days_above_2std.extend(above_2std)
        
        # Storing days below μ-2σ in a list
        below_2std = df.loc[df['Year'] == year]['Return'] < avg_minus_2std[year]
        days_below_2std.extend(below_2std)
    
    # Adding days above and below std_dev to dataframe
    df['Above_2std'] = days_above_2std
    df['Below_2std'] = days_below_2std

    # Get the count of nbr of days above and below (+/-)2*std_returns year wise   
    nbr_of_days_above_2std = df.groupby('Year')['Above_2std'].sum().astype(int)
    nbr_of_days_below_2std = df.groupby('Year')['Below_2std'].sum().astype(int)

    # Calculating % days aboeva nd below (+-)2*std_returns year wise
    pct_days_above_2std = round((nbr_of_days_above_2std / trading_days) * 100 , 2)
    pct_days_below_2std = round((nbr_of_days_below_2std / trading_days) * 100 , 2)
    
        
    # Creating dataframe for values related to Ques-3
    df_3 = pd.DataFrame(columns = ['Year', 'Trading days', 'μ', 'σ', '%days < μ-2σ', '%days > μ+2σ'])
    df_3['Year'] = year_list
    df_3['Trading days'] = list(trading_days)
    df_3['μ'] = list(avg_returns)
    df_3['σ'] = list(std_returns)
    df_3['%days < μ-2σ'] = list(pct_days_below_2std)
    df_3['%days > μ+2σ'] = list(pct_days_above_2std)
    
    # Printing Ques-3 output to console
    print('\nQuestion-3')
    print('**********')
    print(df_3.to_string(index = False))
    
    print('\n*********** Conclusion *******************')
    print('From year 2014-2019: '
          '\nNumber of days when daily returns are greater than μ+2σ: {}'
          '\nNumber of days when daily returns are less than μ-2σ: {}'.format(nbr_of_days_above_2std.sum() , nbr_of_days_below_2std.sum())) 
    
    
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)
        