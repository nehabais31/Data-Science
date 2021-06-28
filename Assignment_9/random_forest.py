"""
Neha Bais
MET CS-677 Assignment-9.2

Implementing Random Forest classifier to predict labels for year 2019

1. take N = (1 to 10) and d = (1 to 5). For each N and f construct RF classifier.
2. Compute error rate for 2019 using 2018 as training data.
3. Find best combination of N and d.
4. Using optimum values from 2018, create Confusion matrix for 2019
5. True +ve and -ve rate for year 2019
6. Implement trading strategy with predicted labels and compare the results with
   buy and hold strategy.
"""

import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from helper_function import *


def random_forest_classifier( X_train, Y_train, X_test, N, d) :
    model = RandomForestClassifier(n_estimators = n, max_depth = d,
                                           criterion = 'entropy')
    model.fit(X_train, Y_train)
    prediction =  model.predict(X_test)
    
    return prediction


ticker = 'BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
ticker_file = os.path.join(input_dir, ticker + '.csv')



try :
    df = pd.read_csv(ticker_file)
    
    # function to extract desired columns
    # extract_df in helper_function script
    final_data = extract_data(df)
    
    # extract 2018 and 2019 data separately
    data_2018 = final_data.loc[final_data['Year'] == 2018].copy()
    data_2019 = final_data.loc[final_data['Year'] == 2019].copy()
    
    #****************** Training 2018 data ***********************************#
    X_18_features = data_2018[['mean_return', 'volatility']].values
    le = LabelEncoder()
    Y_18_class = le.fit_transform(data_2018['Label'].values)
    
    #******************** Test 2019 data ************************************#
    X_19_features = data_2019[['mean_return', 'volatility']].values
    Y_19_class = le.fit_transform(data_2019['Label'].values)
    
    error_rate = {}
    min_error_rate = 1.0   # initialize this as maximum possible error rate value
    
    for d in range(1,6) :
        err_rate_list = []
        for n in range(1, 11) :
            pred_labels = random_forest_classifier( X_18_features, Y_18_class, 
                                                   X_19_features, n, d) 
            
            err_rate_list.append(round( np.mean(pred_labels != Y_19_class), 2))
            
            # Check for optimal n and d values providing minimum error rate
            if err_rate_list[-1] < min_error_rate :
                min_error_rate = err_rate_list[-1]
                best_nd_pair = [n, d]
                best_pred_labels = pred_labels
        
        error_rate[d] = err_rate_list
    
    # Plot error rates for different estimators and deth of tree
    plt.figure( figsize =(10 ,4))
    ax = plt. gca ()
    ax. xaxis.set_major_locator(MaxNLocator( integer = True ))
    for i in range(1,6) :
        plt.plot(np.arange(1,11), error_rate[i], marker = 'o',
                     linestyle ='solid',label = f'max-depth = {i}')
                           
    plt.legend(loc= 'best')
    plt.title ('Random Forest for 2019 label prediction')
    plt.xlabel ('Number of estimators')
    plt.ylabel ('Error Rate')  
    plt.show()
    
    print('\n1. Best combination of N and d -> '
          f'N = {best_nd_pair[0]} ; d = {best_nd_pair[1]}')
    
    accuracy_best_nd = round(np.mean(best_pred_labels == Y_19_class) * 100, 2)
    print(f'Accuracy for 2019 with optimal N and d values -> {accuracy_best_nd}%')
    
    
    ###################################################
    #   Confusion Matrix with optimal N and d values  #
    ###################################################
    
    cf_2019 = confusion_matrix(Y_19_class, best_pred_labels)
    print(f'\n2. Confusion Matrix for 2019 --> \n {cf_2019} \n')
       
    # True +ve and -ve rate for 2019
    tn, fp, fn, tp = cf_2019.ravel()
    
    true_pos_rate_19 = tp / (tp + fn)
    true_neg_rate_19 = tn / (tn + fp)
    
    print(f'3. True positive rate for 2019 = {true_pos_rate_19*100:.2f}%.')
    print(f'4. True negative rate for 2019 = {true_neg_rate_19*100:.2f}%.')
    

    ##############################################################
    #              Trading for 2019 with predicted labels        #
    ##############################################################
    
    data_2019['RF_labels']  = np.where(best_pred_labels == 1, 'Red','Green')
    
    RF_trading = trading_strategy(data_2019, data_2019['RF_labels'])
    buy_n_hold = buy_and_hold(data_2019)
    
    profitable_strategy = 'Random Forest' if RF_trading[len(RF_trading)-1] > \
            buy_n_hold[len(buy_n_hold)-1] else 'Buy & hold'    
    
       
    # Plot portfolio balance for 2 strategies
    plt.figure( figsize =(10 ,4))
    ax = plt. gca ()
    ax. xaxis.set_major_locator(MaxNLocator( integer = True ))
    plt.plot(np.array(list(RF_trading.keys())), np.array(list(RF_trading.values())), 
                color ='red', linestyle ='solid',label = 'Random Forest')
    
    plt.plot(np.array(list(buy_n_hold.keys())), np.array(list(buy_n_hold.values())), 
                color ='green', linestyle ='solid',label = 'Buy & Hold')
                                
    plt.legend()
    plt.title ('2019 - Portfolio balance for 2 trading strategies')
    plt.xlabel ('Week Numbers')
    plt.ylabel ('Portfolio Balance')  
    plt.show()

    print('\nConclusion :')
    print(f'At end of year trading with {profitable_strategy} '
          'resulted in a larger amount. \n'
          f'Portfolio balance for Random Forest at EOY is ${RF_trading[len(RF_trading)-1]} \n'
          f'Portfolio balance for buy & hold at EOY is ${buy_n_hold[len(buy_n_hold)-1]}')

    ##### Portfolio balance at EOY
    # kNN                    -> $148.12   
    # Logistic               -> $148.7 
    # Linear(degree = 3)     -> $125.17 
    # Linear Discriminant    -> $175.13 
    # Quadratic Discriminant -> $146.59 
    # Naive Bayesian         -> $148.21
    # Decision Tree          -> $170.32
    # Buy n Hold             -> $144.2 
    
    #### Accuracy for 2019
    # kNN                    -> 83.02 %   
    # Logistic               -> 81.13 %
    # Linear(degree = 3)     -> 58.49 %  
    # Linear Discriminant    -> 75.47 % 
    # Quadratic Discriminant -> 88.68 % 
    # Naive Bayesian         -> 90.57 %
    # Decision Tree          -> 84.91 %
    
    
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)        

