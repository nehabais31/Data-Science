"""
Neha Bais
MET CS-677 Assignment-8.1

Implementing Gaussian Naive Bayesian 

1. Compute accuracy for 2019
2. Confusion matrix for 2019
3. True +ve and -ve rate for year 2019
4. Implement trading strategy with predicted labels and compare the results with
   buy and hold strategy.
"""

import numpy as np
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import LabelEncoder
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
    Y_18_class = le.fit_transform(data_2018['Label'].values)
    
    #************** 2019 *********************#
    X_19_features = data_2019[['mean_return', 'volatility']].values
    Y_19_class = le.fit_transform(data_2019['Label'].values)
    
    NB_classifier = GaussianNB().fit(X_18_features, Y_18_class)
    prediction = NB_classifier.predict(X_19_features)
    
    # Accuracy for 2019
    accuracy_19 = round(np.mean(prediction == Y_19_class) * 100, 2)
    print('\n\t\t Naive Bayesian classifier')
    print('\t   -----------------------------------\n')
    print(f'1. Accuracy for year 2019 training with year 2018 -> {accuracy_19}% \n')
    
    
    ##############################
    # Confusion matrix for 2019  #
    ##############################
    
    cf_2019 = confusion_matrix(Y_19_class, prediction) # TN = 19 FP = 2  | FN = 3 TP = 29
    print(f'2. Confusion matrrix for 2019 -> \n {cf_2019}\n')
    
    # True positive and true negative rate for 2019
    tn, fp, fn, tp = cf_2019.ravel()
    
    true_pos_rate = tp / (tp + fn)
    true_neg_rate = tn / (tn + fp)
    
    print(f'3. True positive rate for 2019 = {true_pos_rate*100:.2f}%.')
    print(f'4. True negative rate for 2019 = {true_neg_rate*100:.2f}%.')
    
    
    ##############################################################
    #              Trading for 2019 with predicted labesl        #
    ##############################################################
    
    data_2019['NB_labels']  = np.where(prediction == 1, 'Red','Green')
    
    NB_trading = trading_strategy(data_2019, data_2019['NB_labels'])
    buy_n_hold = buy_and_hold(data_2019)
    
    profitable_strategy = 'Naive Bayesian' if NB_trading[len(NB_trading)-1] > \
            buy_n_hold[len(buy_n_hold)-1] else 'Buy & hold'    
    
       
    # Plot portfolio balance for 2 strategies
    plt.figure( figsize =(10 ,4))
    ax = plt. gca ()
    ax. xaxis.set_major_locator(MaxNLocator( integer = True ))
    plt.plot(np.array(list(NB_trading.keys())), np.array(list(NB_trading.values())), 
                color ='red', linestyle ='solid',label = 'Naive Bayesian')
    
    plt.plot(np.array(list(buy_n_hold.keys())), np.array(list(buy_n_hold.values())), 
                color ='green', linestyle ='solid',label = 'Buy & Hold')
                                
    plt.legend()
    plt.title ('2019 - Portfolio balance for 2 trading strategies')
    plt.xlabel ('Week Numbers')
    plt.ylabel ('Portfolio Balance')  
    plt.show()
    
    
    print('\nConclusion :')
    print(f'At end of year {profitable_strategy} resulted in a larger amount. \n'
          f'Portfolio balance for Naive bayesian at EOY is ${NB_trading[len(NB_trading)-1]} \n'
          f'Portfolio balance for buy & hold at EOY is ${buy_n_hold[len(buy_n_hold)-1]}')
    
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)    