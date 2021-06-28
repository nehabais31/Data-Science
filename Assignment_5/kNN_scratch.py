# -*- coding: utf-8 -*-
"""
Neha Bais
MET CS-677 Assignment-4.3

Implementing KNN classifier from scratch fr 3 metrices
- Manhattan p = 1
- Minkovski p = 1.5
- Euclidean p = 2

"""


import numpy as np
import os
import pandas as pd
import matplotlib . pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler , LabelEncoder
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
from helper_function import *


ticker = 'BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
ticker_file = os.path.join(input_dir, ticker + '.csv')
 

def true_pos_neg_rate(cf):
    '''
    Calculate True +ve and -ve rate
    '''
    true_pos = cf[1][1]
    true_neg = cf[0][0]
    
    total_positive = sum(cf[1])   # TP + FN
    total_negative = sum(cf[0])   # TN + FP
    
    sensitivity = true_pos / total_positive
    specificity = true_neg / total_negative
    
    return sensitivity, specificity
 
               
def scaling_feature(data):
    '''
    Scaling our features and class values
    '''
    
    feature_names = ['mean_return', 'volatility']
    
    X = data[feature_names].values
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    le = LabelEncoder ()
    Y = le.fit_transform ( data['Class'].values )
    
    return X ,Y 


def calculate_distance(x1, x2, p):
    return np.linalg.norm(x1-x2 , ord = p)


class custom_kNN :
    def __init__(self, k , p):
        self.k = k
        self.p = p
        
    def __str__(self):
        return "number_neightbors_k = " + str(self.k) + ", p = " + str(self.p)
    
    def fit(self, X, Y):
        '''
        Fit x and y  training sets
        '''
        self.X = X
        self.Y = Y
        
    def predict(self,  x_test):
        '''
        Take each point from testing dataset to compute distance from 
        training dataset
        '''
        predicted_labels = [self._predict(x) for x in x_test]
        return np.array(predicted_labels)
        
    def _predict(self, x):
         # compute distances
         distances = [calculate_distance(x, x_train, self.p) for 
                      x_train in self.X]
         
         # get k nearest samples and labels
         k_indices = np.argsort(distances)[ : self.k]
         k_nearest_labels = [self.Y[i] for i in k_indices]
                 
         # majority vote, most common class label
         most_common = Counter(k_nearest_labels).most_common(1)
         
         return most_common[0][0]
     
        
    def draw_decision_boundary (self , data):
         # Taking 2 weeks => Week_nbr = 18 (Red) & 34 (Green)
        X ,Y = scaling_feature(data)
        
        x_min  = X[np.argmin( X[ : , 0] )][0]
        x_max  = X[np.argmax( X[ : , 0] )][0]
        
        y_min = X[np.argmin( X[ : , 1] )][1]
        y_max = X[np.argmax( X[ : , 1] )][1]
        
          
        # Creating a meshgrid with x_range and y_range [ week_ids = 18 & 34]    
        xs, ys = np.meshgrid(np.linspace(x_min, x_max, 60), 
                            np.linspace(y_min, y_max, 60))
                
        
        #plt.plot(xs,ys, marker='.', color='k', linestyle='none')
    
        self.fit(X, Y)
    
        predictions = self.predict(np.c_[xs.ravel(), ys.ravel()])
        
        predictions = predictions.reshape(xs.shape)
         
        # plot colors for meshpoints            
        for i in range(len(predictions)) :
            for j in range(len(predictions)):
                if predictions[i][j] == 1:
                    plt.scatter(xs[i][j], ys[i][j], c = 'green', s = 50)
                elif predictions[i][j] == 0 :
                    plt.scatter(xs[i][j], ys[i][j], c = 'red',  s = 50)
        
        # Plotting for week 18(Red) and 34(Green)    
        x_week_18 = X[18][0]
        y_week_18 = X[18][1]
            
        x_week_34 = X[34][0]
        y_week_34 = X[34][1]

        plt.scatter(x_week_18, y_week_18, c= 'white',  s = 500)
        plt.scatter(x_week_34, y_week_34, c = 'black',  s = 500)            
                
        plt.title('Decision boundary for kNN predicted labels')        
        plt.xlabel('µ : Average daily returns' )     
        plt.ylabel('σ: Volatility')
        plt.show() 
    
           
try :
    df = pd.read_csv(ticker_file)
    
    # function to extract desired columns
    # extract_df in helper_function script
    final_data = extract_data(df)
    
    # extract 2018 and 2019 data separately
    data_2018 = final_data.loc[final_data['Year'] == 2018].copy()
    data_2019 = final_data.loc[final_data['Year'] == 2019].copy()
    
    data_2018['Class'] = data_2018.loc[:,'Label'].apply(lambda x: 1 
                                                   if x == 'Green' else 0)
        
    X_18 ,Y_18 = scaling_feature(data_2018)
    
    # Split into train and test dataset
    X_train_18 ,X_test_18 , Y_train_18 , Y_test_18 =  \
                            train_test_split (X_18, Y_18 , test_size =0.5 ,
                                              random_state =3)
    
    # Computing best value of k for different metrices
    acc_manhattan_18 = []
    acc_minkovski_18 = []
    acc_euclidean_18 = []
    for k in range (3,12 ,2):
        clf_manhattan_18  = custom_kNN(k, 1)
        clf_minkovski_18 = custom_kNN(k, 1.5)
        clf_euclidean_18  = custom_kNN(k, 2)
        
        clf_manhattan_18.fit(X_train_18, Y_train_18)
        clf_minkovski_18.fit(X_train_18, Y_train_18)
        clf_euclidean_18.fit(X_train_18, Y_train_18)
        
        predictions_man_18 = clf_manhattan_18.predict(X_test_18)
        acc_manhattan_18.append(np.mean ( predictions_man_18 == Y_test_18 ))
    
        predictions_min_18 = clf_minkovski_18.predict(X_test_18)
        acc_minkovski_18.append(np.mean ( predictions_min_18 == Y_test_18 ))
        
        predictions_euc_18 = clf_euclidean_18.predict(X_test_18)
        acc_euclidean_18.append(np.mean ( predictions_euc_18 == Y_test_18 ))
        
        
    # Plotting accuracy for k for 3 metrices
    plt.figure( figsize =(10 ,4))
    ax = plt. gca ()
    ax. xaxis.set_major_locator(MaxNLocator( integer = True ))
    plt.plot(range(3,12 ,2), acc_manhattan_18, color ='red', linestyle ='dashed',
                marker ='o', markerfacecolor ='black', markersize =10,
                label = 'Manhattan - 1')
    plt.plot(range(3,12 ,2), acc_minkovski_18, color ='green', linestyle ='dashed',
                marker ='o', markerfacecolor ='black', markersize =10,
                label = 'Minkovski - 1.5')
    plt.plot(range(3,12 ,2), acc_euclidean_18, color ='blue', linestyle ='dashed',
                marker ='o', markerfacecolor ='black', markersize =10,
                label = 'Euclidean - 2')
    plt.legend()
    plt.title ('Accuracy Rate vs. k for 2018 BAC stock Subset ')
    plt.xlabel ('number of neighbors : k')
    plt.ylabel ('Accuracy Rate ')  
    plt.show()
    
    
    acc_man = round(max(acc_manhattan_18) * 100, 2)
    acc_min = round(max(acc_minkovski_18) * 100, 2)
    acc_euc = round(max(acc_euclidean_18) * 100, 2)
     
    print('\nFor 2018 -->')
    print(f'Euclidean accuracy: {acc_euc:.2f}%'
          f'\nManhattan accuracy: {acc_man:.2f}%'
          f'\nMinkovski accuracy: {acc_min:.2f}%')
    
    print('\nConclusion for 2018: ')
    print('As seen from graph, the optimal value of k for all these 3 metrices = 5.'
          f'\nEuclidean gives the highest accuracy: {max(acc_euclidean_18) * 100 : .2f}%')
    
    
    ########################### 2019 #####################################
    
    data_2019['Class'] = data_2019.loc[:,'Label'].apply(lambda x: 1 
                                                   if x == 'Green' else 0)
    
    X_19 ,Y_19 = scaling_feature(data_2019)
    
          
    '''
    Predicting labels for 2019 with 2018 as training set
    '''
    
     ##############################
    #     Euclidean distance     #
    ##############################                        
    clf_euclidean_19 = custom_kNN(k = 5, p = 2)
    clf_euclidean_19.fit(X_18, Y_18)
    predictions_euc_19 = clf_euclidean_19.predict(X_19)
    
    acc_euclidean_19 = (np.sum(predictions_euc_19 == Y_19) / len(Y_19)) * 100 
    
    
    ##############################
    #     Manhattan distance     #
    ##############################                        
    clf_manhattan_19 = custom_kNN(k = 5, p = 1)
    clf_manhattan_19.fit(X_18, Y_18)
    predictions_man_19 = clf_manhattan_18.predict(X_19)
    
    acc_manhattan_19 = (np.sum(predictions_man_19 == Y_19) / len(Y_19)) * 100 
    
    
    ##############################
    #     Minkovski distance     #
    ##############################                        
    clf_minkovski_19 = custom_kNN(k = 5, p = 1.5)
    clf_minkovski_19.fit(X_18, Y_18)
    predictions_min_19 = clf_minkovski_18.predict(X_19)
    
    acc_minkovski_19 = (np.sum(predictions_min_19 == Y_19) / len(Y_19)) * 100
    
    print('\nFor 2019 -->')
    print(f'Euclidean accuracy: {acc_euclidean_19:.2f}%'
          f'\nManhattan accuracy: {acc_manhattan_19:.2f}%'
          f'\nMinkovski accuracy: {acc_minkovski_19:.2f}%')
    
       
    print('\nConclusion: \nBoth for 2018 and 2019 Euclidean gives the highest'
          ' accuracy rate. However, the accuracy rate is reduced for 2019 as compared to 2018.'
          ' For 2018 it was 92.5% and for 2019 it was reduced to 83.02%.'
          ' Also, for both the years, accuracy is same for Manhattan and Minkovski metrics.')
    
    ####### Confusion matrix for p = 1, 1.5, 2
    
    cf_man = confusion_matrix(Y_19, predictions_man_19)  # TN = 31 FP = 1  | FN = 11 TP = 10
    
    cf_mink = confusion_matrix(Y_19, predictions_min_19)  # TN = 31 FP = 1  | FN = 11 TP = 10
    
    cf_euc = confusion_matrix(Y_19, predictions_euc_19)  # TN = 30 FP = 2  | FN = 7 TP = 14
    
    print(f'\nConfusion matrix for Manhattan:\n {cf_man}'
          f'\nConusion matrix for Minkovski:\n {cf_mink}'
          f'\nConfusion matrix for Euclidean:\n {cf_euc}')
    
    
    ##################################################
    #      Sensitivity & Specificity for 2019        #
    ##################################################
    
    # Manhattan : p = 1
    sensitivity_man, specificity_man = true_pos_neg_rate(cf_man)
    
    # Minkovski : p = 1.5
    sensitivity_mink, specificity_mink = true_pos_neg_rate(cf_mink)
    
    # Euclidian : p = 2
    sensitivity_euc, specificity_euc = true_pos_neg_rate(cf_euc)
    
    print('\nTrue +ve and -ve rate for 3 metrices: \n'
          f'Manhattan: Sensitivity = {sensitivity_man:.2f}  Specificity = {specificity_man:.2f}\n'
          f'Minkovski: Sensitivity = {sensitivity_mink:.2f}  Specificity = {specificity_mink:.2f}\n'
          f'Euclidean: Sensitivity = {sensitivity_euc:.2f}  Specificity = {specificity_euc:.2f}')
    
    print('\nManhattan & MInkovski perform similar. The sensitivity and spcificty are close t each other.\n'
          'Euclidean sensitivity is much greater than the other two ,also the specificity is lower compared to other 2 metrices.')
    
    ##########################################
    #    Plotting the graph  p vs accuracy   #
    ##########################################
    accuracy_values_18 = [acc_man, acc_min, acc_euc]
    accuracy_values_19 = [acc_manhattan_19, acc_minkovski_19, acc_euclidean_19]
    
    p_values = np.array([1 , 1.5, 2]).astype(str)
    #acc_values = accuracy_values
    plt.figure(figsize =(10 ,4))
    ax = plt.gca()
    ax. xaxis.set_major_locator(MaxNLocator( integer = True ))
    plt.plot(p_values, accuracy_values_18, color ='red', linestyle ='dashed',
                marker ='o', markerfacecolor ='black', markersize =10,
                label = '2018')
    plt.plot(p_values, accuracy_values_19, color ='blue', linestyle ='dashed',
                marker ='o', markerfacecolor ='black', markersize =10,
                label = '2019')
    plt.title ('2018 & 2019 - Accuracy Rate vs. distance metrics ')
    plt.xlabel ('Distance metrics : p')
    plt.ylabel ('Accuracy')   
    plt.legend()
    plt.show()
    
    
    #######################################################################
    #         Trading for 2019 with predicted labesl for all metrices     #
    #######################################################################
    
    data_2019['knn_man']  = np.where(predictions_man_19 == 1, 'Green','Red')
    data_2019['knn_mink'] = np.where(predictions_min_19 == 1, 'Green','Red')
    data_2019['knn_euc']  = np.where(predictions_euc_19 == 1, 'Green','Red')
    
    # trade with knn predicted labels
    # function imported from helper package
    
    # Manhattan metric : p = 1
    knn_trading_man = trading_strategy(data_2019, data_2019['knn_man'])
    mean_pred_trading_man = np.array(list(knn_trading_man.values())).mean().round(2)
    
    # Minkovski metric : p = 1.5
    knn_trading_mink = trading_strategy(data_2019, data_2019['knn_mink'])
    mean_pred_trading_mink = np.array(list(knn_trading_mink.values())).mean().round(2)
    
    # Euclidean metric : p = 1
    knn_trading_euc = trading_strategy(data_2019, data_2019['knn_euc'])
    mean_pred_trading_euc = np.array(list(knn_trading_euc.values())).mean().round(2)
    
    
    # buy and hold strategy
    buy_n_hold_19 = buy_and_hold(data_2019)
    
    print('\nTrading Conclusion: \n'
          'Euclidean strategy reulted in largest portfolio balance at the end of year.'
          'Balance at the EOY for Euclidean is $148.12. Manhattan and Minkovski both resulted in similar'
          ' balance of $146.94. However, all f these 3 strategies perform better than Buy and Hold strategy.')
    
    
    ##############################################
    #              Decision Boundary             #
    ##############################################

    clf = custom_kNN(k = 5, p = 1.5)
    clf.draw_decision_boundary ( data_2019 )
           

except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)           