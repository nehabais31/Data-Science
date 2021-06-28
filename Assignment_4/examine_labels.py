# -*- coding: utf-8 -*-
"""
Neha Bais
MET CS-677 Assignment - 4.1

Compare the labels for both years and find whether 
nearest neighbors approach could be a good (or bad) classier.

"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from helper_function import *


ticker = 'BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
ticker_file = os.path.join(input_dir, ticker + '.csv')
output_2018 = os.path.join(input_dir, ticker + '_reduced_data_2018.csv')
output_2019 = os.path.join(input_dir, ticker + '_reduced_data_2019.csv')

    
def plot_labels(label, mean, volatality, week_id, week_return, year):
    '''
    Function to plot labels 
    based on mean and volatality of daily returns 
    on a weekly basis.
    '''
    for i in range(len(label)) :
        if label[i] == 'Green' :
            plt.scatter(mean[i], volatality[i], c = 'green',
                        s = (10 if abs(week_return[i]) <= 2.0 else 80))
        elif label[i] == 'Red' :
            plt.scatter(mean[i], volatality[i], c = 'red',
                        s = (10 if abs(week_return[i]) <= 2.0 else 80))
    
    # Adding week ids to each point
    for i, txt in enumerate(week_id):
        plt.annotate(txt, (mean[i],volatality[i]))
        
    plt.title('{} Label based on µ and σ'.format(year))        
    plt.xlabel('µ : Average daily returns' )     
    plt.ylabel('σ: Volatility')
    plt.show()    


try :
    df = pd.read_csv(ticker_file)
    
    # function to extract desired columns
    final_data = extract_data(df)
    
    # extract 2018 and 2019 data separately
    data_2018 = final_data.loc[final_data['Year'] == 2018].copy()
    data_2019 = final_data.loc[final_data['Year'] == 2019].copy()
    
    # creating lists of mean, volatility, weekly return and labels from dataframe
    # for 2018
    mean_2018 = data_2018['mean_return'].to_list()
    volatality_2018 = data_2018['volatility'].to_list()
    label_2018 = data_2018['Label'].to_list()
    week_2018 = data_2018['Week_Number'].to_list()
    week_return_2018 = calculate_weekly_return(data_2018)
    
    # final dataframe with desired values
    data_2018 = data_2018.assign(Weekly_Return = week_return_2018)
    #data_2018.to_csv(output_2018, index = False)
    
     # for 2019
    mean_2019 = data_2019['mean_return'].to_list()
    volatality_2019 = data_2019['volatility'].to_list()
    label_2019 = data_2019['Label'].to_list()
    week_2019 = data_2019['Week_Number'].to_list()
    week_return_2019 = calculate_weekly_return(data_2019)
    
    # final dataframe with desired values
    data_2019 = data_2019.assign(Weekly_Return = week_return_2019)
    data_2019.to_csv(output_2019, index = False)
    
    # plot labels for 2018
    plot_labels(label_2018, mean_2018, volatality_2018, week_2018,week_return_2018, 2018)
    
    # plot labels for 2019
    plot_labels(label_2019, mean_2019, volatality_2019, week_2019,week_return_2019, 2019)
    
    # Conclusion 
    print('\nConclusion : \n For 2018: '
          '\n1. The green points are clustered at the area where the volatility is low '
          'and the average daily return is positive.\n'
          '2. As the volatility increases the size of red points increases.\n'
          '3. The maximum volatility observed for green points is 1.5.\n'
          '4. Also, the size of green points increases with µ.\n'
          '5. The green points are clusterd to a specific area, while the red points are scattered.\n'
          '6. The pattern is somewhat similar for both the years, except for 2019, I observed that,'
              ' few of the green points are also there where the mean returns is -ve..\n'
          '7. I expect that the nrearest neighbour classifier trained for 2018 will work for 2019 as well.')
           
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)    