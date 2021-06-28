"""
@author: Neha Bais
MET CS677 - Assignment 2.1

Create a file with mean returns and volatility 
group by week_nbr for each year.

"""

# run this  !pip install pandas_datareader
from pandas_datareader import data as web
import os
import math
import numpy as np 
import pandas as pd


ticker='BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
output_file = os.path.join(input_dir, ticker + '_weekly_return_volatility.csv')

try:
    df = web.get_data_yahoo(ticker, start='2014-01-01',end='2019-12-31')
    df['Return'] = df['Adj Close'].pct_change()
    df['Return'].fillna(0, inplace = True)
    df['Return'] = 100.0 * df['Return']
    df['Return'] = df['Return'].round(3)        
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])
    df['Week_Number'] = df['Date'].dt.strftime('%U')
    df['Year'] = df['Date'].dt.year 
    df['Year_Week'] = df['Date'].dt.strftime('%Y-%U')
    df_2 = df[['Year', 'Year_Week','Week_Number' , 'Return']]
    df_2.index = range(len(df))
    df_grouped = df_2.groupby(['Year', 'Week_Number'])['Return'].agg([np.mean, np.std])
    df_grouped.reset_index(['Year', 'Week_Number'], inplace=True)
    df_grouped.rename(columns={'mean': 'mean_return', 'std':'volatility'}, inplace=True)
    df_grouped.fillna(0, inplace=True)
    df_grouped.to_csv(output_file, index=False)
#    df_grouped_2 = df_grouped.fillna(0)
#    df_grouped_2.to_csv(output_file, index=False)
    
except Exception as e:
    print(e)