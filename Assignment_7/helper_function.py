"""
Neha Bais
MET CS-677 

Helper functions for Assignment-4

"""

import pandas as pd
import numpy as np
import os

def extract_data(df) :
    '''
    Extract data for 2018 and 2019
    Returned columns: 
        Year_Week           
        Year                  
        Week_Number           
        Week_Start_Date    
        Label                 
        Week_End_Date     
        End_Adj_Close        
        mean_return           
        volatility 
    '''
    
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
    
    df_2 = df[['Year', 'Year_Week','Week_Number' , 'Return']]
    df_grouped = df_2.groupby(['Year', 'Week_Number'])['Return'].agg([np.mean, np.std])
    df_grouped.reset_index(['Year', 'Week_Number'], inplace=True)
    df_grouped.rename(columns={'mean': 'mean_return', 'std':'volatility'}, inplace=True)
    df_grouped.fillna(0, inplace=True)
    
    avg_vol_data = df_grouped.loc[(df_grouped['Year'] == 2018) | (df_grouped['Year'] == 2019)].copy()
    
    data = pd.merge(week_data, avg_vol_data, on=['Year','Week_Number'])
    data['mean_return'] =  ((data['mean_return'] * 100 ).round(3)).to_list()
    data['volatility'] = ((data['volatility'] * 100).round(3)).to_list()
    
    return data


def calculate_weekly_return(data) :
    '''
    Function to calculate weekly return based on 
    Adj Close price
    
    Input : dataframe 
    retrns : list containing weekly returns
    
    '''
    # calculating weekly return
    weekly_return = []
    
    for i in range(len(data)) :
        prev_fri_close = data.iloc[i-1]['End_Adj_Close']
        curr_fri_close = data.iloc[i]['End_Adj_Close']
        
        if i == 0 :
            weekly_return.append(0)
        else :
            return_rate = (curr_fri_close - prev_fri_close) / prev_fri_close
            weekly_return.append(round((return_rate * 100), 3))
            
    return weekly_return 



def trading_strategy(data, label) :
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
        if label.iloc[i] == 'Red':
            invest_amnt = 100.00
            weekly_balance[data.iloc[i]['Week_Number']] = invest_amnt
            continue
        
        elif label.iloc[i] == 'Green' :
            invest_amnt = 100.00
            shares = round(invest_amnt / data.iloc[i]['Start_Adj_close'] , 2)
            current_label = 'Green'
            current_bal = shares * data.iloc[i]['End_Adj_Close']
            weekly_balance[data.iloc[i]['Week_Number']] = round(current_bal , 2)
            start_pos = i
            break
        
        
    # Iterate through rest of data  
    for i in range(start_pos, len(data) - 1) :
        next_week_label = label.iloc[i+1]
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