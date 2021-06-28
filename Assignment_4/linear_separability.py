"""
Neha Bais
MET CS-677 Assignment - 4.2

1. Examine 2018 plot and draw a line separating green & red points
2. Compute equation of this line
3. Using this line assign labels for 2019
4. Trade for 2019 based on these new labels

"""



import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from helper_function import *


ticker = 'BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
ticker_file = os.path.join(input_dir, ticker + '.csv')


def assign_labels_19(data, slope, intercept):
    '''
    Predicting labels for 2019 based on the line 
    classifer that is calculated
    
    y = 2.14 * x + 0.93
    
    Return : dataframe including predicted labels 
             and predicted sigma columns
    
    '''
    predicted_sigma = []
    predicted_label = []
    
    for i in range(len(data)) :
        mu_1 = data.iloc[i]['mean_return'].round(2)
        sigma_1 = data.iloc[i]['volatility'].round(2)
        
        pred_sigma = round((mu_1 * slope + intercept) , 2)
        predicted_sigma.append(pred_sigma)
    
        if pred_sigma <= sigma_1:
            predicted_label.append('Green')
        else:
            predicted_label.append('Red')
    
    # adding new predicted columns to our dataframe        
    data = data.assign(predicted_volatility = predicted_sigma, 
                       predicted_label = predicted_label)  
    
        
        
    return data


def plot_labels(label, mean, volatality, week_id, week_return, year, slope, intercept):
    '''
    Function to plot labels 
    based on mean and volatality of daily returns 
    on a weekly basis.
    '''
    for i in range(len(label)) :
        if label[i] == 'Green' :
            plt.scatter(mean[i], volatality[i], c = 'green',
                        s = (10 if abs(week_return[i]) <= 2.0 else 50))
        elif label[i] == 'Red' :
            plt.scatter(mean[i], volatality[i], c = 'red',
                        s = (10 if abs(week_return[i]) <= 2.0 else 50))
    
    # Adding week ids to each point
    for i, txt in enumerate(week_id):
        plt.annotate(txt, (mean[i],volatality[i]))
        
    x = np.array(mean)
    y = slope*x + intercept
    #y = 2.14*x + 0.93    # equation of line
    
    plt.plot(x,y,'-b')
    plt.grid()    
    plt.title('{} Label based on µ and σ'.format(year))        
    plt.xlabel('µ : Average daily returns' )     
    plt.ylabel('σ: Volatility')
    plt.show()    


try :
    df = pd.read_csv(ticker_file)
    
    #***************** Extracting data for 2018 & 2019 ***********************#
    
    # function to extract desired columns
    # extract_df in helper_function script
    final_data = extract_data(df)
    
    # extract 2018 and 2019 data separately
    data_2018 = final_data.loc[final_data['Year'] == 2018].copy()
    data_2019 = final_data.loc[final_data['Year'] == 2019].copy()
    
    # calculate weekly return defined in helper_funtion
    week_return_2018 = calculate_weekly_return(data_2018)
    week_return_2019 = calculate_weekly_return(data_2019)
    
    # final dataframe with desired values
    data_2018 = data_2018.assign(Weekly_Return = week_return_2018)
    data_2019 = data_2019.assign(Weekly_Return = week_return_2019)
    
    
    #************** Dropping some points to plot a line **********************#
    
    # reducing dataset so as to form a line to separate 2 colors points
    data_2018 = data_2018.drop([6,15,16,22,24,27,28,46,47], axis = 0)
    
    # for 2018
    mean_2018 = data_2018['mean_return'].to_list()
    volatality_2018 = data_2018['volatility'].to_list()
    label_2018 = data_2018['Label'].to_list()
    week_2018 = data_2018['Week_Number'].to_list()
    week_return_2018 = data_2018['Weekly_Return'].to_list()
    
    # selecting 2 points to calculate eq of line
    x1, x2 = 0.5, -0.2
    y1, y2 = 2.0 , 0.5
    
    slope = round((y2 - y1) / (x2 - x1) , 2)
    intercept = round(y1 - (slope * x1) , 2)
    
    # equation of line 
    #y = 2.14*x + 0.93
    
    plot_labels(label_2018, mean_2018, volatality_2018, week_2018,
                week_return_2018, 2018, slope, intercept)
    
    
    #******************* Predicting 2019 labels with classifier **************#
    # predicting labels for 2019
    pred_data_19 = assign_labels_19(data_2019, slope, intercept)
    
    
    #******************** Trading with predicted labels for 2019 *************#
    
    # implement trading startegy based on predicted labels
    trading_prediction_19 = trading_strategy(pred_data_19, pred_data_19['predicted_label'])
    
    
    # Calculating avergae and volatility of weekly balance 
    mean_pred_trading_19 = np.array(list(trading_prediction_19.values())).mean().round(2)
    volatility_pred_trading_19 = np.array(list(trading_prediction_19.values())).std().round(2)
    
    print('\nConclusion: ')
    
    print('\nOur classifier that separates green & red points is: y = 2.14*x + 0.93')
    
    print('\nWith predicted labels for 2019:\n'
          f'Average of portfolio balance: ${mean_pred_trading_19}')
    
    print(f'Volatility of portfolio balance: {volatility_pred_trading_19}')
    
    print('\nThe previous trading where we did trading based on our assigned labels gave more profit '
          'as compared to this one. The portfolio average balance was around $119 earlier and I was in profit. '
          '\nHowever, with this label prediction I faced a loss and the portfolio average balance dropeed to $76. '
          'I found that in this case, Buy & Hold is much profitable compared to trading with this predicted labelling.')
    
    #For 2019  with our assigned labels
    #Average of weekly balance:  119.33 
    #Volatility of weekly balance:  16.88
    
    
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)       