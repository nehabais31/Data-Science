"""
Neha Bais
MET CS-677 Assignment-6.1a

Implementing KNN & Logistic classifier using SHAPLEY

1. Compute marginal cntributions selecting a single feature set 
   for both knn and logistic regression.
"""

import numpy as np
import os
import pandas as pd
import matplotlib . pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from helper_function import *


ticker = 'BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
ticker_file = os.path.join(input_dir, ticker + '.csv')


def scaling_feature(features, labels):
    '''
    Splitting our test and training data
        Returns: X_train ,X_test , Y_train , Y_test
    '''
    scaler = StandardScaler()
    scaler.fit(features)
    X = scaler.transform(features)
    le = LabelEncoder ()
    Y = le.fit_transform ( labels )
    return X ,Y 

def kNN_classifier(X_18, Y_18, X_19, Y_19) :
    '''
    Returns accuracy by predicting
    labels based on kNN classifier
    '''
    # Predicting labels with k = 5
    knn_classifier = KNeighborsClassifier ( n_neighbors = 5)
    knn_classifier.fit ( X_18 , Y_18 )
    pred_k_19 = knn_classifier.predict ( X_19 )
    accuracy = round(np.mean ( pred_k_19 == Y_19 ) * 100 , 2)
    
    return accuracy

def logistic_regression(X_18, Y_18, X_19, Y_19) :
    '''
    REturns accuracy by predicting labels
    based on logistic regression
    '''
    # Logistic classifier 
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit( X_18, Y_18)
    prediction = log_reg_classifier.predict(X_19)
    accuracy = round(np.mean(prediction == Y_19) * 100, 2)
    
    return accuracy


try :
    df = pd.read_csv(ticker_file)
    
    # function to extract desired columns
    # extract_df in helper_function script
    final_data = extract_data(df)
    
    # extract 2018 and 2019 data separately
    data_2018 = final_data.loc[final_data['Year'] == 2018].copy()
    data_2019 = final_data.loc[final_data['Year'] == 2019].copy()
    
    #****************** Training 2018 data ***********************************#
    
    data_2018['Class'] = data_2018.loc[:,'Label'].apply(lambda x: 1 
                                                   if x == 'Green' else 0)
    X_18_features = data_2018[['mean_return', 'volatility']].values
    Y_18_features = data_2018['Class'].values
        
    X_18 ,Y_18 = scaling_feature(X_18_features, Y_18_features)
    
   
    #************** 2019 *********************#
          
    # Splitting into test and training dataset
    data_2019['Class'] = data_2019.loc[:,'Label'].apply(lambda x: 1 
                                                   if x == 'Green' else 0)
    
    X_19_features = data_2019[['mean_return', 'volatility']].values
    Y_19_features = data_2019['Class'].values
    X_19 ,Y_19 = scaling_feature(X_19_features, Y_19_features)
       
    # kNN & Logistic classifier
    knn_accuracy_19 = kNN_classifier(X_18, Y_18, X_19, Y_19)
    log_accuracy_19 = logistic_regression(X_18, Y_18, X_19, Y_19)
    #print(f'\nAccuracy for 2019 with k = 5 is {knn_accuracy_19}%.')
    
    
    ##########################################################################
    #                  Shapely Feature Implementation                        # 
    ##########################################################################
    
    ####################################
    # Selecting mu as the only feature #
    ####################################
    X_18_first = data_2018['mean_return'].values.reshape(-1,1)
    X_mu_18 ,Y_mu_18 = scaling_feature(X_18_first, Y_18_features)
     
    X_19_first = data_2019['mean_return'].values.reshape(-1,1)                                       
    X_mu_19 , Y_mu_19 = scaling_feature(X_19_first, Y_19_features)
    
    # kNN & Logistic Classifier
    knn_accuracy_mu_19 = kNN_classifier(X_mu_18, Y_mu_18, X_mu_19, Y_mu_19)
    log_accuracy_mu_19 = logistic_regression(X_mu_18, Y_mu_18, X_mu_19, Y_mu_19)
 
    # Marginal for kNN and logistic 
    knn_marginal_mu = round( knn_accuracy_19 - knn_accuracy_mu_19 , 2)
    log_marginal_mu = round( log_accuracy_19 - log_accuracy_mu_19 , 2)
    
    
    ############################################
    # Selecting volatility as the only feature #
    ############################################
    X_18_second = data_2018['volatility'].values.reshape(-1,1)
    X_sigma_18 ,Y_sigma_18 = scaling_feature(X_18_second, Y_18_features)
     
    X_19_second = data_2019['volatility'].values.reshape(-1,1)                                       
    X_sigma_19 , Y_sigma_19 = scaling_feature(X_19_second, Y_19_features)
    
    # kNN & Logistic classifier
    knn_accuracy_sigma_19 = kNN_classifier(X_sigma_18, Y_sigma_18, X_sigma_19, 
                                           Y_sigma_19)
    log_accuracy_sigma_19 = logistic_regression(X_sigma_18, Y_sigma_18, X_sigma_19, 
                                           Y_sigma_19)

    # Marginal for kNN and logistic        
    knn_marginal_sigma = round(knn_accuracy_19 - knn_accuracy_sigma_19 , 2)
    log_marginal_sigma = round(log_accuracy_19 - log_accuracy_sigma_19 , 2)
      
    # Printing output to console
    # Accuracy dataframe
    accuracy_data = {
             'kNN': [knn_accuracy_19, knn_accuracy_mu_19, knn_accuracy_sigma_19],
             'Logistic': [log_accuracy_19, log_accuracy_mu_19, log_accuracy_sigma_19]}
                            
    df_accuracy = pd.DataFrame(accuracy_data, 
                      columns = ['kNN', 'Logistic'],
                      index = ['All', 'mu', 'sigma'])
    
    # Marginal dataframe
    marginal_data = {
             'kNN': [ knn_marginal_mu, knn_marginal_sigma],
             'Logistic': [ log_marginal_mu, log_marginal_sigma]}
    
    df_marginal = pd.DataFrame(marginal_data, 
                      columns = ['kNN', 'Logistic'],
                      index = [ 'mu', 'sigma'])
    
    print('\nAccuracy for mu and sigma as the only feature value')
    print(df_accuracy)
    
    print('\nMarginal for mu and sigma as the only feature value')
    print(df_marginal)
    

    print('\nConclusion:')
    print('The marginal contribution is comparativley smaller when I selected sigma as the '
          'only feature as compared to the one when mu is selected as the only feature, for both '
          'kNN and Logistic classifier.'
          'So, in this case, selecting sigma as the only feature will give us a good prediction.'
          ' Sigma is an important feature of my stock.')
    
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)       