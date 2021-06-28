# -*- coding: utf-8 -*-
"""
Neha Bais
MET CS-677 Assignment-5

Plot decision boundary for 2019
using weeks 18(red) 34(green)

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


def scaling_feature(data):
    '''
    Scaling X and Y values
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
     
        
    def draw_decision_boundary (self , X, Y ):
         # Taking 2 weeks => Week_nbr = 18 (Red) & 34 (Green)
        x_min  = X[18][0]
        x_max  = X[34][0]
        
        y_min = X[18][1]
        y_max = X[34][1]
        
          
        # Creating a meshgrid with x_range and y_range [ week_ids = 18 & 34]    
        xs, ys = np.meshgrid(np.linspace(x_min, x_max, 50), 
                            np.linspace(y_min, y_max, 50))
        
        
        #plt.plot(xs,ys, marker='.', color='k', linestyle='none')
    
        self.fit(X, Y)
    
        predictions = self.predict(np.c_[xs.ravel(), ys.ravel()])
    
        predictions = predictions.reshape(xs.shape)
         
        for i in range(len(predictions)) :
            for j in range(len(predictions)):
                if predictions[i][j] == 1:
                    plt.scatter(xs[i][j], ys[i][j], c = 'green', s = 10)
                elif predictions[i][j] == 0 :
                    plt.scatter(xs[i][j], ys[i][j], c = 'red',  s = 10)
    
        #plotting 2 points of week id 18 and 34
        #x_week_18 =             
        
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
    #data_2018 = final_data.loc[final_data['Year'] == 2018].copy()
    data_2019 = final_data.loc[final_data['Year'] == 2019].copy()
    
    data_2019['Class'] = data_2019.loc[:,'Label'].apply(lambda x: 1 
                                                   if x == 'Green' else 0)
        
    X_19 ,Y_19 = scaling_feature(data_2019)
    
    clf = custom_kNN(k = 5, p = 1.5)
    clf.draw_decision_boundary ( X_19, Y_19 )
    
    
    
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)     