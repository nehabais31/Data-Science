"""
Neha Bais
MET CS-677 Assignment-9.3

Implementing Adaboost with 3 classifiers of choice to predict labels for year 2019

learning rate = 0.5 and 1
weak learners(N) = range(1 to 15)

1. Plot 2019 error rates for each N 
2. Fr each base estimatr find best N for learning rate = 0.5
3. Accuracy for each base estimator for best N
4. what classier is best to use as base estimator for your data?
5. Implement trading strategy with predicted labels(Adaboost with best estimtor)
   and compare the results with buy and hold strategy.
"""

import numpy as np
import os
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import LabelEncoder, StandardScaler
from helper_function import *


def evaluate_model(clf, l_rate, x_train, y_train, x_test, y_test) :
    error_rate = {}
    prediction = {}
    
    for n in range(1,16) :
        model = AdaBoostClassifier(n_estimators = n, base_estimator = clf, 
                                     learning_rate = l_rate)
        model.fit(x_train, y_train)
        prediction[n] = model.predict(x_test)
        error_rate[n] = round(np.mean(model.predict(x_test) != y_test) , 4)
    
    return prediction, error_rate


def plot_error_rates(err_rate_l1, err_rate_l2, title ) :
    plt.figure( figsize =(10 ,4))
    ax = plt. gca ()
    ax. xaxis.set_major_locator(MaxNLocator( integer = True ))
    plt.plot(range(1,16), np.array(list(err_rate_l1.values())), marker = 'o', 
                color ='red', linestyle ='solid',label = 'rate = 0.5')
    
    plt.plot(range(1,16), np.array(list(err_rate_l2.values())), marker = 'o',
                color ='green', linestyle ='solid',label = 'rate = 1')
                                
    plt.legend()
    plt.title (title)
    plt.xlabel ('Number of estimators')
    plt.ylabel ('Error Rate')  
    plt.show()


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
    
    
    # Base classsifiers for AdaBoost
    log_reg = LogisticRegression()
    naive_bayesian = GaussianNB()
    svc = SVC(probability = True, kernel = 'linear')
    
    #classifiers = [log_reg, naive_bayesian, random_forest]
    classifiers = [log_reg, naive_bayesian, svc]
           
    for clf in classifiers :
        # Logistic Regression boosting
        if clf == log_reg :
            # for learning rate = 0.5
            log_pred_l1, log_err_rate_l1 = evaluate_model(clf, 0.5, 
                                            X_18_features, Y_18_class, 
                                            X_19_features, Y_19_class)
                
            # for learning rate = 1
            log_pred_l2, log_err_rate_l2 = evaluate_model(clf, 1, 
                                            X_18_features, Y_18_class, 
                                            X_19_features, Y_19_class)
            # plot the graph for 2 learning rates
            plot_error_rates(log_err_rate_l1, log_err_rate_l2, 
                             'Logistic Regression Adaboost' )
            
            # finding best n for learning rate = 0.5
            log_best_n = min(log_err_rate_l1.items(), key = lambda x: x[1])
            
        # Naive Bayesian boosting 
        elif clf == naive_bayesian :
            # for learning rate = 0.5
            nb_pred_l1, nb_err_rate_l1 = evaluate_model(clf, 0.5, 
                                            X_18_features, Y_18_class, 
                                            X_19_features, Y_19_class)
                
            # for learning rate = 1
            nb_pred_l2, nb_err_rate_l2 = evaluate_model(clf, 1, 
                                            X_18_features, Y_18_class, 
                                            X_19_features, Y_19_class)
            
            # plot the graph for 2 learning rates
            plot_error_rates(nb_err_rate_l1, nb_err_rate_l2, 
                             'Naive Bayesian Adaboost' )
            
            # finding best n for learning rate = 0.5
            bayesian_best_n = min(nb_err_rate_l1.items(), key = lambda x: x[1])
        
        # Support Vector classifier 
        elif clf == svc :
            # for learning rate = 0.5
            svc_pred_l1, svc_err_rate_l1 = evaluate_model(clf, 0.5, 
                                            X_18_features, Y_18_class, 
                                            X_19_features, Y_19_class)
                
            # for learning rate = 1
            svc_pred_l2, svc_err_rate_l2 = evaluate_model(clf, 1, 
                                            X_18_features, Y_18_class, 
                                            X_19_features, Y_19_class)
                
            # plot the graph for 2 learning rates
            plot_error_rates(svc_err_rate_l1, svc_err_rate_l2, 
                             'Support Vector Adaboost' )
            
            # finding best n for learning rate = 0.5
            svc_best_n = min(svc_err_rate_l1.items(), key = lambda x: x[1])    
          
      
    print('\n\t\t Adaboost classifier')  
    print('\t   -----------------------------------\n')
    print('2. Best value of N for learning rate = 0.5 --> \n'
          f'   Logistic Regression = {log_best_n[0]} \n'
          f'   Naive Bayesian      = {bayesian_best_n[0]} \n'
          f'   Support Vector      = {svc_best_n[0]} \n')     
        
    
    #####################################################     
    #    Accuracy for each base estimator for best N    #
    #####################################################
    log_best_pred = log_pred_l2[log_best_n[0]]
    log_accuracy = round(np.mean(log_best_pred == Y_19_class) * 100 , 2)
    
    nb_best_pred = nb_pred_l2[bayesian_best_n[0]]
    bayesian_accuracy = round(np.mean(nb_best_pred == Y_19_class) * 100 , 2)
    
    svc_best_pred = svc_pred_l2[svc_best_n[0]]
    svc_accuracy = round(np.mean(svc_best_pred == Y_19_class) * 100 , 2)
    
    acc_data = pd.DataFrame({'Models': ['Logistic Regression' , 'Naive Bayesian',
                                       'Support Vector'],
                            'Accuracy': [log_accuracy, bayesian_accuracy, 
                                         svc_accuracy]})
    
    print('3. Accuracy for each base estimator using best N (learning rate = 0.5 --> \n')
    print(acc_data.to_string(index=False))
    
    
    # Finding best estimator 
    best_estimator = acc_data.iloc[acc_data['Accuracy'].argmax()]
    print(f'\n4. Best estimator for my data -->\n   {best_estimator[0]} '
          f'with an accuracy of {best_estimator[1]} %.')
    
    
    ##############################################################
    #              Trading for 2019 with predicted labesl        #
    ##############################################################
    # Decoding labels for best estimator for trading 
    
    if best_estimator[0] == 'Logistic Regression' :
        data_2019['Ada_labels']  = np.where(log_best_pred == 1, 'Red','Green')
    elif best_estimator[0] == 'Naive Bayesian' :
        data_2019['Ada_labels']  = np.where(nb_best_pred == 1, 'Red','Green')
    elif best_estimator[0] ==  'Support Vector' :
        data_2019['Ada_labels']  = np.where(svc_best_pred == 1, 'Red','Green')
    
    
    Ada_trading = trading_strategy(data_2019, data_2019['Ada_labels'])
    buy_n_hold = buy_and_hold(data_2019)
    
    profitable_strategy = f'Adaboost with {best_estimator[0]} as base classifier' \
                           if Ada_trading[len(Ada_trading)-1] > \
                           buy_n_hold[len(buy_n_hold)-1] else 'Buy & hold'    
    
    
    print(f'\n5. Trading using Adaboost with best estimator: {best_estimator[0]}')
    
    # Plot portfolio balance for 2 strategies
    plt.figure( figsize =(10 ,4))
    ax = plt. gca ()
    ax. xaxis.set_major_locator(MaxNLocator( integer = True ))
    plt.plot(np.array(list(Ada_trading.keys())), np.array(list(Ada_trading.values())), 
                color ='red', linestyle ='solid',label = f'Adaboost -{best_estimator[0]}')
    
    plt.plot(np.array(list(buy_n_hold.keys())), np.array(list(buy_n_hold.values())), 
                color ='green', linestyle ='solid',label = 'Buy & Hold')
                                
    plt.legend()
    plt.title ('2019 - Portfolio balance for 2 trading strategies')
    plt.xlabel ('Week Numbers')
    plt.ylabel ('Portfolio Balance')  
    plt.show()
    
    
    print('\nConclusion :')
    print(f'At end of year {profitable_strategy} resulted in a larger amount. \n'
          f'Portfolio balance for {profitable_strategy} at EOY is ${Ada_trading[len(Ada_trading)-1]} \n'
          f'Portfolio balance for buy & hold at EOY is ${buy_n_hold[len(buy_n_hold)-1]}')
    
    
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)      