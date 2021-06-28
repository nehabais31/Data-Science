"""
Neha Bais
MET CS-677 Assignment-8.1

Implementing Linear and Quadratic Discriminant

1. Compute accuracy for 2019
2. Confusion matrix for 2019
3. True +ve and -ve rate for year 2019
4. Implement trading strategy with predicted labels and compare the results with
   buy and hold strategy for both linear and quadratic discriminant.
"""

import numpy as np
import os
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import LabelEncoder, StandardScaler
from helper_function import *


def compute_confusion_matrix(actual, prediction):
    '''
    Returns confusion matrix
            True +ve rate
            True -ve rate
    '''
    cf = confusion_matrix(actual, prediction) # TN  FP   | FN  TP 
        
    # True positive and true negative rate 
    tn, fp, fn, tp = cf.ravel()
    
    true_pos_rate = tp / (tp + fp)
    true_neg_rate = tn / (tn + fn)
    
    return cf, true_pos_rate, true_neg_rate


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
    
    #************** Training 2018 data ********************#
    X_18_features = data_2018[['mean_return', 'volatility']].values
    scaler = StandardScaler().fit(X_18_features)
    X_18_features = scaler.transform(X_18_features)
    
    le = LabelEncoder()
    Y_18_class = le.fit_transform(data_2018['Label'].values)
    
    #************** Testing 2019 data *********************#
    X_19_features = data_2019[['mean_return', 'volatility']].values
    Y_19_class = le.fit_transform(data_2019['Label'].values)
    scaler = StandardScaler().fit(X_19_features)
    X_19_features = scaler.transform(X_19_features)
    
    
    ###########################
    #   Linear Discriminant   #
    ###########################
    lda_classifier = LDA()
    lda_classifier.fit(X_18_features, Y_18_class)
    
    lda_prediction = lda_classifier.predict(X_19_features)
    lda_accuracy = round(np.mean(lda_prediction == Y_19_class) * 100, 2)
    print('\n\t\t Discriminant Analysis')
    print('\t   -----------------------------------\n')
    print(f'1. Linear Discriminant accuracy for 2019 -> {lda_accuracy}% \n')
    
    
    ##############################
    #   Quadratic Discriminant   #
    ##############################
    qda_classifier = QDA()
    qda_classifier.fit(X_18_features, Y_18_class)
    
    qda_prediction = qda_classifier.predict(X_19_features)
    qda_accuracy = round(np.mean(qda_prediction == Y_19_class) * 100, 2)
    
    print(f'2. Quadratic Discriminant accuracy for 2019 -> {qda_accuracy}% \n')
    
    better_classifier = 'Linear Discriminant is better' if lda_accuracy > qda_accuracy \
        else 'Quadratic Discriminant is better. \n'
        
    print(better_classifier)   
    
    
    ##############################
    # Confusion matrix for 2019  #
    ##############################
    
    # For Linear discriminant
    cf_lda, lda_true_pos_rate, lda_true_neg_rate =  \
                    compute_confusion_matrix(Y_19_class, lda_prediction) # TN = 19 FP = 2  | FN = 3 TP = 29
    
    print(f'3. Confusion matrix for LDA -> \n {cf_lda}\n')
    print(f'4. True positive rate for LDA = {lda_true_pos_rate*100:.2f}%.')
    print(f'   True negative rate for LDA = {lda_true_neg_rate*100:.2f}%.\n')
    
    # For Quadratic discriminant
    cf_qda, qda_true_pos_rate, qda_true_neg_rate =  \
                    compute_confusion_matrix(Y_19_class, qda_prediction) # TN = 19 FP = 2  | FN = 3 TP = 29
    
    print(f'3. Confusion matrix for QDA -> \n {cf_qda}\n')
    print(f'4. True positive rate for QDA = {qda_true_pos_rate*100:.2f}%.')
    print(f'   True negative rate for QDA = {qda_true_neg_rate*100:.2f}%.\n')
    
    
    
    ##############################################################
    #              Trading for 2019 with predicted labesl        #
    ##############################################################
    
    data_2019['LDA_labels']  = np.where(lda_prediction == 1, 'Red','Green')
    data_2019['QDA_labels']  = np.where(qda_prediction == 1, 'Red','Green')
    
    LDA_trading = trading_strategy(data_2019, data_2019['LDA_labels'])
    QDA_trading = trading_strategy(data_2019, data_2019['QDA_labels'])
    
    buy_n_hold = buy_and_hold(data_2019)
    
    lda_profitable_strategy = 'Linear Discriminant' if \
            LDA_trading[len(LDA_trading)-1] > \
            buy_n_hold[len(buy_n_hold)-1] else 'Buy & hold'    
            
    qda_profitable_strategy = 'Quadratic Discriminant' if \
            QDA_trading[len(QDA_trading)-1] > \
            buy_n_hold[len(buy_n_hold)-1] else 'Buy & hold'   
    
       
    # Plot portfolio balance for 2 strategies
    
    plt.figure( figsize =(10 ,4))
    ax = plt. gca ()
    ax. xaxis.set_major_locator(MaxNLocator( integer = True ))
    plt.plot(np.array(list(LDA_trading.keys())), np.array(list(LDA_trading.values())), 
                color ='red', linestyle ='solid',label = 'Linear Discriminant')
    
    plt.plot(np.array(list(QDA_trading.keys())), np.array(list(QDA_trading.values())), 
                color ='blue', linestyle ='solid',label = 'Quadratic Discriminant')
    
    plt.plot(np.array(list(buy_n_hold.keys())), np.array(list(buy_n_hold.values())), 
                color ='green', linestyle ='solid',label = 'Buy & Hold')
                                
    plt.legend()
    plt.title ('2019 - Portfolio balance for 3 trading strategies')
    plt.xlabel ('Week Numbers')
    plt.ylabel ('Portfolio Balance')  
    plt.show()
    
    
    print('\nConclusion :')
    print('Comparing the performance with Buy and Hold strategy, both Linear ' 
          'and Quadratic discriinant resulted in a larger portfolio balance at ' 
          'end of year. \n')
    print('Portfolio balance at EOY -> \n'
          f'Linear Discriminant    -> ${LDA_trading[len(LDA_trading)-1]} \n'
          f'Quadratic Discriminant -> ${QDA_trading[len(QDA_trading)-1]} \n'
          f'Buy n Hold             -> ${buy_n_hold[len(buy_n_hold)-1]} \n'
          '\nAlso, if we compare both the discriminants, trading with labels predicted '
          'by Linear discriminant is more profitable in my case.')
       
    
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)  

    