"""
Neha Bais
MET CS-677 Assignment-11

Implementing k-means clustering algorithm

1. Take k = range(1,9) and compute the distortion vs. k. 
   Use the "knee" method to find out the best k.
2. Find % of red and green weeks for each cluster using best k.
3. Find any pure clusters.
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from helper_function import *


ticker = 'BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
ticker_file = os.path.join(input_dir, ticker + '.csv')


def get_scaled_data(data):
    feature_names = ['mean_return', 'volatility']
    
    X = data[feature_names].values
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    le = LabelEncoder ()
    Y = le.fit_transform ( data['Label'].values )
    
    return X ,Y 



try :
    df = pd.read_csv(ticker_file)
    
    # function to extract desired columns
    # extract_df in helper_function script
    final_data = extract_data(df)
      
    # Scaling features and label encodeing    
    X_features, Y_labels = get_scaled_data(final_data)
    
        
    kmeans_classifier = KMeans(n_clusters = 3)
    y_kmeans = kmeans_classifier.fit_predict(X_features)
    centroids = kmeans_classifier.cluster_centers_
    
    
    fig , ax = plt.subplots (1, figsize =(7 ,5))
    plt.scatter (X_features[y_kmeans == 0, 0], X_features[y_kmeans == 0, 1],
                   s = 75, c = 'green', label = 'Green week ')
    plt.scatter (X_features[ y_kmeans == 1, 0], X_features[ y_kmeans == 1, 1],
                   s = 75, c = 'red', label = 'Red week')
    
    plt.scatter ( centroids [:, 0], centroids [: ,1] ,
                   s = 200 , c = 'black', label = 'Centroids ')
    x_label = 'mu'
    y_label = 'sigma'
    plt.legend()
    plt.xlabel( x_label )
    plt.ylabel( y_label )
    plt.tight_layout()
    plt.show()
    
    
    #######################
    #  Finding optimal k  #
    #######################
    inertia_list = []
    for k in range(1,9):
        kmeans_classifier = KMeans(n_clusters = k)
        y_kmeans = kmeans_classifier.fit_predict(X_features)
        inertia = kmeans_classifier.inertia_
        inertia_list.append(inertia)
        
    fig , ax = plt.subplots(1, figsize =(7 ,5))
    plt.plot(range(1, 9) , inertia_list , marker ='o', color ='green')
    plt.xlabel('number of clusters : k')
    plt.ylabel('inertia')
    plt.title('Distortion vs. Number of clusters')
    plt.tight_layout()
    plt.grid(linestyle = '--')
    plt.show()
    
    print('\nThe best k = 5')
    
    
    #################################
    # Using best k examine clusters #
    #################################
    kmeans_classifier = KMeans(n_clusters = 5)
    kmeans_classifier.fit(X_features)
    pred = kmeans_classifier.predict(X_features)
        
    # Creating a dataframe for features, labels and clusters
    cluster_df = final_data.filter(['mean_return','volatility','Label'],
                                   axis = 1)
    cluster_df['Cluster'] = pred
    
    print('\nCount of assigned points to cluster --')
    print(cluster_df.groupby('Cluster')['Label'].value_counts())
    
    #######################################################
    # Getting % of green and red points in each cluster   #
    #######################################################
    label_pct = pd.crosstab(
                cluster_df['Cluster'], cluster_df['Label']).apply(
                    lambda x: round(x / x.sum() * 100, 2), axis =1)
    
    print('\nQuestion - 2')
    print(f'Percentage of green & red labels in each cluster: \n\n {label_pct}')  
    
    ##########################
    # Finding pure clusters  #
    ##########################
    pure_red_cluster = label_pct[label_pct['Red'] > 90.00].index.tolist()
    pure_green_cluster = label_pct[label_pct['Green'] > 90.00].index.tolist()
    
    print('\nQuestion - 3')
    if len(pure_red_cluster) != 0 :
        print(f'Pure clusters with Red weeks more than 90% : {pure_red_cluster}')
    
    if len(pure_green_cluster) != 0 : 
        print(f'Pure clusters with Green weeks more than 90% : {pure_green_cluster}')
    
    
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)           
    