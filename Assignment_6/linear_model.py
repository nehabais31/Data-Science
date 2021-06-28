# -*- coding: utf-8 -*-
"""
Neha Bais
MET CS-677 Assignment-6.2

Implementing Linear model
1. Take w = [5 to 12] and degree = [1,2,3] for 2018 data and construct polynomials.
2. Use these polynomials to predict weekly labels.
3. For each degree take best w and predict labels for 2019.
4. Compute confusion matrices for each degree for 2019.
5. Trading strategies for each d with best w from 2018 data.
"""


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from helper_function import *


ticker = 'BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
ticker_file = os.path.join(input_dir, ticker + '.csv')


def true_pos_neg(cf) :
    '''
    Calculates true positive and negative rate
    Input : Confusion matrix
    '''
    true_pos = cf[1][1]
    true_neg = cf[0][0]
    
    total_positive = sum(cf[1])   # TP + FN
    total_negative = sum(cf[0])   # TN + FP
    
    tpr = round( true_pos / total_positive , 2)
    tnr = round( true_neg / total_negative , 2)
    
    return tpr, tnr

def calculate_prediced_price(d, w, x, y):
    '''
    Linear model fitting 
    Input : degree, window, week_id and adj_close 
    Returns : Predicted labels
    '''
    labels = []
    for i in range(len(x)-w) :
        weights = np.polyfit(x[i : i+w], y[i : i+w], d)
        model = np.poly1d(weights)
        predicted = model(x[i+w])
        if predicted > y[i+w-1] :
            labels.append(1)
        else :
            labels.append(0)
    
    return labels    
        
try :
    df = pd.read_csv(ticker_file)
    
    # function to extract desired columns
    # extract_df in helper_function script
    final_data = extract_data(df)
    
    # extract 2018 and 2019 data separately
    data_2018 = final_data.loc[final_data['Year'] == 2018].copy()
    data_2019 = final_data.loc[final_data['Year'] == 2019].copy()
    
    ########################################################
    #                 2018 - Label Prediction              #
    ########################################################
    
    label_2018 = data_2018.loc[:,'Label'].apply(lambda x: 1 
                                                   if x == 'Green' else 0)
    
    week_id_18   = np.array(data_2018['Week_Number'])
    adj_close_18 = np.array(data_2018['End_Adj_Close'])
    
    window = np.arange(5,13) #[i for i in range(5,13)]
    degree = [1,2,3]
    
    accuracy_1 = {}
    accuracy_2 = {}
    accuracy_3 = {}
    
    for d in degree :
        for w in window :
            if d == 1:
                pred_labels = calculate_prediced_price(d, w, week_id_18, adj_close_18)
                accuracy_1[w] = round(np.mean(pred_labels == 
                                      label_2018[w:len(label_2018)]) * 100, 2)
                
            if d == 2:
                pred_labels = calculate_prediced_price(d, w,  week_id_18, adj_close_18)
                accuracy_2[w] = round(np.mean(pred_labels == 
                                        label_2018[w:len(label_2018)])* 100, 2)
                
            if d == 3:
                pred_labels = calculate_prediced_price(d, w, week_id_18, adj_close_18)
                accuracy_3[w] = round(np.mean(pred_labels == 
                                        label_2018[w:len(label_2018)])* 100, 2)    
     
    print('\n\tAccuracy for 2018 for 3 degrees and different w\n')    
    print(' Degree-1 ->' ,accuracy_1,'\n', 
          'Degree-2 ->', accuracy_2, '\n',
          'Degree-3 ->',accuracy_3)
    
    # Plotting accuracy for k for 3 degrees
    plt.figure( figsize =(10 ,4))
    ax = plt.gca()
    ax. xaxis.set_major_locator(MaxNLocator( integer = True ))
    plt.plot(window, np.array(list(accuracy_1.values())), color ='red', linestyle ='dashed',
                marker ='o', markerfacecolor ='black', markersize =10,
                label = 'Degree - 1')
    plt.plot(window, np.array(list(accuracy_2.values())), color ='green', linestyle ='dashed',
                marker ='o', markerfacecolor ='black', markersize =10,
                label = 'Degree - 2')
    plt.plot(window, np.array(list(accuracy_3.values())), color ='blue', linestyle ='dashed',
                marker ='o', markerfacecolor ='black', markersize =10,
                label = 'Degree - 3')
    plt.legend()
    plt.title ('Accuracy Rate vs. w for 2018 BAC stock Subset ')
    plt.xlabel ('Window Size : w')
    plt.ylabel ('Accuracy Rate ')  
    plt.show()
    
    # Degree- 1 : w = 7
    # Degree- 2 : w = 12
    # Degree- 3 : w = 7
    
    ########################################################
    #                 2019 - Label Prediction              #
    ########################################################
    
    label_2019 = data_2019.loc[:,'Label'].apply(lambda x: 1 
                                                   if x == 'Green' else 0)
    
    week_id_19   = np.array(data_2019['Week_Number'])
    adj_close_19 = np.array(data_2019['End_Adj_Close'])
    
    # For Degree-1 ; w = 7
    w_1 = 7
    y_value_1 = np.concatenate( (adj_close_18[-w_1: len(adj_close_18)], 
                                 adj_close_19))
    x_value_1 = np.arange(0, len(y_value_1))
    pred_1 = calculate_prediced_price(1, w_1, x_value_1, y_value_1)
    accuracy_19_d1 = round(np.mean(pred_1 == label_2019)* 100, 2)    
   
    # For Degree-2 ; w = 12
    w_2 = 12
    y_value_2 = np.concatenate( (adj_close_18[-w_2: len(adj_close_18)], 
                                 adj_close_19))                                          
    x_value_2 = np.arange(0, len(y_value_2))
    pred_2 = calculate_prediced_price(2, w_2, x_value_2, y_value_2)
    accuracy_19_d2 = round(np.mean(pred_2 == label_2019)* 100, 2)   
    
    
    # For Degree-3 ; w = 7
    w_3 = 7
    y_value_3 = np.concatenate( (adj_close_18[-w_3: len(adj_close_18)], 
                                 adj_close_19))                                          
    x_value_3 = np.arange(0, len(y_value_3))   
    pred_3 = calculate_prediced_price(3, w_3, x_value_3, y_value_3)    
    accuracy_19_d3 = round(np.mean(pred_3 == label_2019)* 100, 2)                              
    
    print('\nAccuracy for 2019:\n',
          f'Degree-1 w-7  : {accuracy_19_d1}% \n'
          f' Degree-2 w-12 : {accuracy_19_d2}% \n',
          f'Degree-3 w-7  : {accuracy_19_d3}% \n')
    
    
    ###############################
    #      Confusion Matrices     #
    ###############################
    cf_d1 = confusion_matrix(label_2019, pred_1)  # TN = 30 FP = 2  | FN = 2 TP = 19
    cf_d2 = confusion_matrix(label_2019, pred_2)  # TN = 30 FP = 2  | FN = 2 TP = 19
    cf_d3 = confusion_matrix(label_2019, pred_3)  # TN = 30 FP = 2  | FN = 2 TP = 19
    
    sensitivity_d1, specificity_d1 = true_pos_neg(cf_d1)
    sensitivity_d2, specificity_d2 = true_pos_neg(cf_d2)
    sensitivity_d3, specificity_d3 = true_pos_neg(cf_d3)
    
    print('\nConfusion matrices for 2019 :\n'
          f'Degree-1 : {cf_d1}\n'
          f'Degree-2 : {cf_d2}\n'
          f'Degree-3 : {cf_d3}')
    
    print(f'\nDegree-1 : Sensitivity - {sensitivity_d1}  Specificity - {specificity_d1}\n'
          f'Degree-2 : Sensitivity - {sensitivity_d2}  Specificity - {specificity_d2}\n'
          f'Degree-3 : Sensitivity - {sensitivity_d3}  Specificity - {specificity_d3}')
 
    
    ##########################################
    #         2019 Trading for 3 degrees     # 
    ##########################################
    
    data_2019['label_degree_1'] = np.where(np.array(pred_1) == 1, 'Green','Red')
    data_2019['label_degree_2'] = np.where(np.array(pred_2) == 1, 'Green','Red')
    data_2019['label_degree_3'] = np.where(np.array(pred_3) == 1, 'Green','Red')
       
    
    trading_degree_1 = trading_strategy(data_2019, data_2019['label_degree_1'])
    mean_trading_d1 = np.array(list(trading_degree_1.values())).mean().round(2)
    
    trading_degree_2 = trading_strategy(data_2019, data_2019['label_degree_2'])
    mean_trading_d2 = np.array(list(trading_degree_2.values())).mean().round(2)
    
    trading_degree_3 = trading_strategy(data_2019, data_2019['label_degree_3'])
    mean_trading_d3 = np.array(list(trading_degree_3.values())).mean().round(2)
    
    print('\n\t\tTrading Conclusion ')
    print('Average Portfolio balance for 2019: \n'
          f'Degree-1 : ${mean_trading_d1} \n'
          f'Degree-2 : ${mean_trading_d2} \n'
          f'Degree-3 : ${mean_trading_d3} \n')
    
    
    print('Trading with degree 3 is more profitable as compared to the other two. '
          'On comparing these 3 trading strategies, degree-1 resulted in lowest portfolio balance at the end of year. '
          'Degree-3 resulted in the highest portfolio balance.')
    
    
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)      