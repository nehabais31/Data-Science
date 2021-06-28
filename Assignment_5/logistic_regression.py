# -*- coding: utf-8 -*-
"""
Neha Bais
MET CS-677 Assignment-4.3

Implementing Logistic Regression classifier

1. Compute equation for logistic regression from 2018 data.
2. Compute accuracy for 2019 with 2018 data as training set.
3. Confusion matrix for 2019.
4. Compute sensitivity & specificity for 2019.
5. Trade based on these new labels for 2019 & compare with buy n hold.

"""



import numpy as np
import os
import pandas as pd
import matplotlib . pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
from helper_function import *


ticker = 'BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
ticker_file = os.path.join(input_dir, ticker + '.csv')


def scaling_feature(data):
    '''
    Splitting our test and training data
    
    Returns: X_train ,X_test , Y_train , Y_test
    
    '''
    
    feature_names = ['mean_return', 'volatility']
    
    X = data[feature_names].values
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    le = LabelEncoder ()
    Y = le.fit_transform ( data['Class'].values )
    
    return X ,Y 

        
try :
    df = pd.read_csv(ticker_file)
    
    # function to extract desired columns
    # extract_df in helper_function script
    final_data = extract_data(df)
    
    # extract 2018 and 2019 data separately
    data_2018 = final_data.loc[final_data['Year'] == 2018].copy()
    data_2019 = final_data.loc[final_data['Year'] == 2019].copy()
    
    # Scaling our test and train data
    # 2018 training data
    data_2018['Class'] = data_2018.loc[:,'Label'].apply(lambda x: 1 
                                                   if x == 'Green' else 0)
    
    x_train_18 , y_train_18 = scaling_feature(data_2018)
    
    # 2019 testing data
    data_2019['Class'] = data_2019.loc[:,'Label'].apply(lambda x: 1 
                                                   if x == 'Green' else 0)
    
    x_test_19 , y_test_19 = scaling_feature(data_2019)
    
    
    # Logistic classifier 
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit( x_train_18, y_train_18)
    
    prediction = log_reg_classifier.predict(x_test_19)
    accuracy = np.mean(prediction == y_test_19)
        
    ####################################### 
    #         Regression equation         #
    #######################################
    
    print('Question-1')
    # calculating weights 
    # w0 = intercept, w1,w2 = coefficients -- mu and sigma value
    w0 = log_reg_classifier.intercept_[0]
    coef = log_reg_classifier.coef_
    
    w1 = coef[0][0]
    w2 = coef[0][1]
    
    # equation of logistic regression from 2018 data
    print('Equation of logistic regression from 2018 data')
    print('y = ' + str(round(w0, 4)) + ' + (' + str(\
        round(w1, 4)) + ' * mu) + (' + str(\
        round(w2, 4)) + ' * sigma)')
    
    
    print('\nQuestion-2')
    print(f'Accuracy for 2019: {accuracy *100: .2f}%')
    
    
    ##########################################
    #     Confusion matrix for 2019          #
    ##########################################
    
    print('\nQuestion-3')
    cf = confusion_matrix(y_test_19, prediction)  # TN = 31 FP = 1  | FN = 9 TP = 12
    print('\nConfusion matrix for 2019: \n', cf)
    
    true_pos = cf[1][1]
    true_neg = cf[0][0]
    
    print('\nQuestion-4')
    total_positive = sum(cf[1])   # TP + FN
    total_negative = sum(cf[0])   # TN + FP
    
    sensitivity = true_pos / total_positive
    specificity = true_neg / total_negative
    
    print(f'True +ve rate predicted for 2019: {sensitivity:.2f}')
    print(f'True -ve rate predicted for 2019: {specificity:.2f}')
    
    
    #############################################################
    #         Trading for 2019 with predicted labels            #
    #############################################################
    
    data_2019['logit_label'] = np.where(prediction == 1, 'Green','Red')
    
    # trade with knn predicted labels
    # function imported from helper package
    logit_trading_19 = trading_strategy(data_2019, data_2019['logit_label'])
    mean_pred_trading_19 = np.array(list(logit_trading_19.values())).mean().round(2)
    
    # buy and hold strategy
    buy_n_hold_19 = buy_and_hold(data_2019)
    
    print('\nConclusion: ')
    print('\nWith predicted labels for 2019:\n'
          f'Average of portfolio balance: ${mean_pred_trading_19}')
    
    print('\nAt the end of the year, trading with labels predicted by logistic regression is more'
          ' profitable as compared to buy and hold strategy.'
          f'\nLogistic startegy resulted in an amount of ${logit_trading_19[52]}'
          f' while buy and hold resulted in ${buy_n_hold_19[52]}.'
          ' Also, I did not find any larger difference in portfolio balance at end of year '
          'between labels predicted by kNN and logistic regression. kNN resulted in $148.12 and logistic '
           'resulted in $148.7 at end of year.')
    
    
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)      