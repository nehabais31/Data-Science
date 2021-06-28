"""
Neha Bais
MET CS-677 Assignment-9.1

Implementing Decision Tree classifier to predict labels for year 2019

1. Compute accuracy for 2019
2. Confusion matrix for 2019
3. True +ve and -ve rate for year 2019
4. Implement trading strategy with predicted labels and compare the results with
   buy and hold strategy.
"""

import numpy as np
import os
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from helper_function import *


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
    Y_18_class = le.fit_transform( data_2018['Label'].values )
    
    X_18_train, X_18_test, Y_18_train, Y_18_test = train_test_split(X_18_features, 
                                    Y_18_class, test_size = 0.3, random_state=3)
    
    tree_classifier = tree.DecisionTreeClassifier(criterion = 'entropy')
    tree_classifier.fit(X_18_train, Y_18_train)
    
    prediction_18 = tree_classifier.predict(X_18_test)
    accuracy_18 = np.mean(prediction_18 == Y_18_test)
    
    
    ###################################################################
    #                    2019  -- Decision Tree                       #
    ###################################################################
    
    X_19_features = data_2019[['mean_return' , 'volatility']].values
    Y_19_class = le.fit_transform(data_2019['Label'].values)
    
    tree_classifier.fit(X_18_features, Y_18_class)
    
    prediction_19 = tree_classifier.predict(X_19_features)
    accuracy_19 = round(np.mean(prediction_19 == Y_19_class) * 100, 2)
    
    print('\n\t\t Decision Tree classifier')
    print('\t   -----------------------------------\n')
    print(f'1. Accuracy for year 2019 -> {accuracy_19}% \n')
    
    ######################
    #   Confusion Matrix #
    ######################
    
    cf_2019 = confusion_matrix(Y_19_class, prediction_19)
    print('\n2. Confusion Matrix for 2019 -->')
    sns.heatmap(cf_2019, annot = True, cmap = 'Blues')
    
    # True +ve and -ve rate for 2019
    tn, fp, fn, tp = cf_2019.ravel()
    
    true_pos_rate_19 = tp / (tp + fn)
    true_neg_rate_19 = tn / (tn + fp)
    
    print(f'3. True positive rate for 2019 = {true_pos_rate_19*100:.2f}%.')
    print(f'4. True negative rate for 2019 = {true_neg_rate_19*100:.2f}%.')
    
    
    ##############################################################
    #              Trading for 2019 with predicted labels        #
    ##############################################################
    
    data_2019['DT_labels']  = np.where(prediction_19 == 1, 'Red','Green')
    
    DT_trading = trading_strategy(data_2019, data_2019['DT_labels'])
    buy_n_hold = buy_and_hold(data_2019)
    
    profitable_strategy = 'Decision Tree' if DT_trading[len(DT_trading)-1] > \
            buy_n_hold[len(buy_n_hold)-1] else 'Buy & hold'    
    
       
    # Plot portfolio balance for 2 strategies
    plt.figure( figsize =(10 ,4))
    ax = plt. gca ()
    ax. xaxis.set_major_locator(MaxNLocator( integer = True ))
    plt.plot(np.array(list(DT_trading.keys())), np.array(list(DT_trading.values())), 
                color ='red', linestyle ='solid',label = 'Decision Tree')
    
    plt.plot(np.array(list(buy_n_hold.keys())), np.array(list(buy_n_hold.values())), 
                color ='green', linestyle ='solid',label = 'Buy & Hold')
                                
    plt.legend()
    plt.title ('2019 - Portfolio balance for 2 trading strategies')
    plt.xlabel ('Week Numbers')
    plt.ylabel ('Portfolio Balance')  
    plt.show()
    
    print('\nConclusion :')
    print(f'At end of year trading with labels predicted by {profitable_strategy} '
          'resulted in a larger amount. \n'
          f'Portfolio balance for Decision Tree at EOY is ${DT_trading[len(DT_trading)-1]} \n'
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