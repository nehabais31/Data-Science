# -*- coding: utf-8 -*-
"""
Neha Bais
MET CS-677 Assignment-3.3

1. Trading Strategy with Labels for 2018 & 2019

2. Buy and Hold Strategy for 2018 & 2019

"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt



ticker = 'BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
ticker_file = os.path.join(input_dir, ticker + '.csv')


def trading_strategy(data) :
    # if next week label = Green and you have no position buy shares worth of $100
    # if next week label = Green and you have position - do nothing
    # if next week label = Red and you have position sell shares at the adj close price at the end of week
    # if next week label = Red and you have no position - do nothing
    
    invest_amnt = 0.00
    shares = 0.00
    current_label = 'Green'
    weekly_balance = {}     # to track weekly stock balance at nd of each week
    
    # First day of trading; Do trading only if first week is marked as Green
    for i in range(len(data) - 1) :
        # When Red do nothing; stay in cash 
        if data.iloc[i]['Label'] == 'Red':
            invest_amnt = 100.00
            weekly_balance[data.iloc[i]['Week_Number']] = invest_amnt
            continue
        
        elif data.iloc[i]['Label'] == 'Green' :
            invest_amnt = 100.00
            shares = round(invest_amnt / data.iloc[i]['Start_Adj_close'] , 2)
            current_label = 'Green'
            current_bal = shares * data.iloc[i]['End_Adj_Close']
            weekly_balance[data.iloc[i]['Week_Number']] = round(current_bal , 2)
            start_pos = i
            break
        
        
    # Iterate through rest of data  
    for i in range(start_pos, len(data) - 1) :
        next_week_label = data.iloc[i+1]['Label']
        #adj_close_start = data.iloc[i+1]['Start_Adj_close']
        adj_close_end = data.iloc[i+1]['End_Adj_Close']
        next_week_nbr = data.iloc[i+1]['Week_Number']

        # Buy shares when current week label = Red and next week label = Green  
        if next_week_label == 'Green' :
            
            if current_label == 'Green' :
                current_bal = shares * adj_close_end
                weekly_balance[next_week_nbr] = round(current_bal , 2)
                
            elif current_label == 'Red':
                shares = current_bal / data.iloc[i]['End_Adj_Close']
                current_bal = 0.00
                weekly_balance[next_week_nbr] = round((shares * adj_close_end),2)
        
        # Sell shares if current week = Green and next week = Red        
        elif next_week_label == 'Red' :
            
            if current_label == 'Red' :
                #current_bal = shares * adj_close_end
                weekly_balance[next_week_nbr] = round((current_bal),2)
                
            elif current_label == 'Green' :
                current_bal = shares * data.iloc[i]['End_Adj_Close']
                shares = 0.00
                weekly_balance[next_week_nbr] = round(current_bal,2)
        
        current_label = next_week_label
        
    return weekly_balance


def buy_and_hold(data) :
    '''
    Buy Stock one first day and sell on lasts day.

    '''
    invest_amnt = 100.00
    
    # Buy shares at first day 
    shares = round((invest_amnt / data.iloc[0]['Start_Adj_close']) , 2)
    weekly_return = {}
    
    for i in range(len(data)) :
        adj_close_end = data.iloc[i]['End_Adj_Close']
        
        weekly_return[i] = round((shares * adj_close_end) , 2)
        
    return weekly_return    



def plot_graph(data_label, data_b_h, year, mean, volatility) :
    '''
    Plot a graph comparing Buy & hold vs Label trading strategy
    '''
    # Plotting label trading and buy & hold vs week days
    x_values = list(data_label.keys())
    plt.title('{} - Label trading vs Buy & Hold strategy'.format(year))
    plt.plot(x_values, list(data_label.values()), c = 'b', 
             label = 'Label Trading\n' + r'$\mu=$' + str(mean) +  
         ', ' + r'$\sigma=$' + str(volatility))
    plt.plot(x_values, list(data_b_h.values()), c = 'g', label = 'Buy & Hold')
    plt.legend()
    plt.show()



try :
    df = pd.read_csv(ticker_file)
    
    # Get week's first day values in a dataframe
    data_start = df[['Year_Week','Year','Week_Number','Date','Adj Close','Label']
                    ].groupby('Year_Week').first().reset_index()
    week_start = data_start.loc[(data_start['Year'] == 2018) | (data_start['Year'] == 2019)].copy()
    week_start.rename(columns={'Date' : 'Week_Start_Date', 'Adj Close' : 'Start_Adj_close'}, inplace = True) 
    
    # Get week's last day data in a dataframe
    data_end = df[['Year_Week','Year','Week_Number','Date','Adj Close', 'Label']
                  ].groupby('Year_Week').last().reset_index()
    week_end = data_end.loc[(data_end['Year'] == 2018) | (data_end['Year'] == 2019)].copy()
    week_end.rename(columns={'Date' : 'Week_End_Date' , 'Adj Close' : 'End_Adj_Close'}, inplace = True) 
    
    # Merge start and end week data into a dataframe
    week_data = pd.merge(week_start, week_end, on=['Year_Week','Year','Week_Number','Label'])
    
    label_trading_2018 = trading_strategy(week_data[week_data['Year'] == 2018])
    label_trading_2019 = trading_strategy(week_data[week_data['Year'] == 2019])
        
    # Buy and Hold Strategy
    trading_buy_hold_18 = buy_and_hold(week_data[week_data['Year'] == 2018])
    trading_buy_hold_19 = buy_and_hold(week_data[week_data['Year'] == 2019])
    
    # Average and volatility of weekly balances
    mean_label_trading_18 = np.array(list(label_trading_2018.values())).mean().round(2)
    mean_label_trading_19 = np.array(list(label_trading_2019.values())).mean().round(2)
    
    volatility_label_trading_18 = np.array(list(label_trading_2018.values())).std().round(2)
    volatility_label_trading_19 = np.array(list(label_trading_2019.values())).std().round(2)
    
    #plotting graph
    plot_graph(label_trading_2018, trading_buy_hold_18, 2018,
               mean_label_trading_18, volatility_label_trading_18)
    plot_graph(label_trading_2019, trading_buy_hold_19, 2019, 
               mean_label_trading_19, volatility_label_trading_19)
      
    
    #---------- Question -1 --------------
    print('\n\t\tQuestion-1')
    print('For 2018 \nAverage of weekly balance: ', mean_label_trading_18 ,
          '\nVolatility of weekly balance: ', volatility_label_trading_18 )
    
    print('\nFor 2019 \nAverage of weekly balance: ', mean_label_trading_19 ,
          '\nVolatility of weekly balance: ', volatility_label_trading_19 )
    
    #---------Question -3 ----------------------
    print('\n\t\tQuestion-3')
    max_balance_18 = max(label_trading_2018.keys(), key = lambda k: label_trading_2018[k])
    min_balance_18 = min(label_trading_2018.keys(), key = lambda k: label_trading_2018[k])
    
    max_balance_19 = max(label_trading_2019.keys(), key = lambda k: label_trading_2019[k])
    min_balance_19 = min(label_trading_2019.keys(), key = lambda k: label_trading_2019[k])
    
    
    print('For 2018\n Maximum account balance was in week {} with ${}'
          .format(max_balance_18, label_trading_2018[max_balance_18]))
    print(' Minimum account balance was in week {} with ${}'
          .format(min_balance_18, label_trading_2018[min_balance_18]))
    
    print('\nFor 2019\n Maximum account balance was in week {} with ${}'
          .format(max_balance_19, label_trading_2019[max_balance_19]))
    print(' Minimum account balance was in week {} with ${}'
          .format(min_balance_19, label_trading_2019[min_balance_19]))
    
    #------------Question-4--------------
    print('\n\t\tQuestion-4')
    print('For 2018:  Final account balance: $', list(label_trading_2018.values())[-1] )
    print('For 2019:  Final account balance: $', list(label_trading_2019.values())[-1] )
    
    #----------Question - 5---------
    print('\n\t\tQuestion-5')
    print('For 2018: \nMaximum duration in which account was growing was between 43rd and 52nd week.\n'
          'The account was in profit for each week. Did not experience a loss as per current labelling startegy.')
      
    print('\nFor 2019: \nMaximum duration in which account was growing was between 41st and 52nd week.\n'
          'The account was in profit for each week.Did not experience a loss as per current labelling strategy.')
    
    
    
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)      