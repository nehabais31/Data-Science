"""
Neha Bais
MET CS-677 Assignment-4.3

Implementing KNN classifier

1. Compute accuracy for kNN classifier for 2018 & find optimal k.
2. Use optimal k from 2018 and predict labels for 2019 & compute accuracy.
3. Use optimal k from 2018 and compute confusion matrix for 2019.
4. Compute sensitivity & specificity for 2019.
5. Trade based on these new labels for 2019 & compare with buy n hold.

"""

import numpy as np
import os
import pandas as pd
import matplotlib . pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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
    
    
    '''
    Calculating accuracy rate and checking for optimal k for 2018
    
    First splitting our dataset into test and train data
    X will contain our feature values [mean_return & volatiltiy]
    Y will contain Label for prediction
    
    '''
    
    #****************** Training 2018 data ***********************************#
    
    
    
    data_2018['Class'] = data_2018.loc[:,'Label'].apply(lambda x: 1 
                                                   if x == 'Green' else 0)
        
    X_18 ,Y_18 = scaling_feature(data_2018)
    
    # Split into train and test dataset
    X_train_18 ,X_test_18 , Y_train_18 , Y_test_18 =  \
                            train_test_split (X_18, Y_18 , test_size =0.5 ,
                                              random_state =3)
    
    accuracy_rate_18 = []

    for k in range (3,12 ,2):
        knn_classifier = KNeighborsClassifier ( n_neighbors =k)
        knn_classifier.fit ( X_train_18 , Y_train_18 )
        pred_k_18 = knn_classifier.predict ( X_test_18 )
        accuracy_rate_18.append(np.mean ( pred_k_18 == Y_test_18 ))
    
    # plotting k vs accuracy for 2018   
    plt.figure( figsize =(10 ,4))
    ax = plt. gca ()
    ax. xaxis.set_major_locator(MaxNLocator( integer = True ))
    plt.plot(range(3,12 ,2), accuracy_rate_18, color ='red', linestyle ='dashed',
                marker ='o', markerfacecolor ='black', markersize =10)
    plt.title ('Accuracy Rate vs. k for 2018 BAC stock Subset ')
    plt.xlabel ('number of neighbors : k')
    plt.ylabel ('Accuracy Rate ')
    
    # Conclusion for 2018
    print(f'\nOptimal value of k for 2018 : 5.\
          \nAccuracy for 2018 with k = 5 is {max(accuracy_rate_18)*100:.2f}%.')
          
          
    #************** Predicting label for 2019 with k= 5 *********************#
          
    # Splitting into test and training dataset
    data_2019['Class'] = data_2019.loc[:,'Label'].apply(lambda x: 1 
                                                   if x == 'Green' else 0)
    
    X_19 ,Y_19 = scaling_feature(data_2019)
    
                         
    # Predicting labels with k = 5
    knn_classifier = KNeighborsClassifier ( n_neighbors = 5)
    knn_classifier.fit ( X_18 , Y_18 )
    pred_k_19 = knn_classifier.predict ( X_19 )
    accuracy_rate_19 = np.mean ( pred_k_19 == Y_19 )
    
    print(f'\nAccuracy for 2019 with k = 5 is {accuracy_rate_19 *100:.2f}%.')
    
    
    #******************* Confusion matrix for 2019 with k = 5 ****************#
    y_true_19 = Y_19
    
    cf = confusion_matrix(y_true_19, pred_k_19)  # TN = 30 FP = 2  | FN = 2 TP = 19
    
    true_pos = cf[1][1]
    true_neg = cf[0][0]
    
    total_positive = sum(cf[1])   # TP + FN
    total_negative = sum(cf[0])   # TN + FP
    
    sensitivity = true_pos / total_positive
    specificity = true_neg / total_negative
    
    print(f'\nConfusion matrix for 2019:\n {cf}')
    print(f'\nTrue +ve rate predicted for 2019: {sensitivity:.2f}')
    print(f'True -ve rate predicted for 2019: {specificity:.2f}')
    
    print('\nThis shows that our kNN classifier predicted more Red labels' 
          ' correctly than Green labels for 2019.')
    
    
    #****************** Trading based on new labels **************************#
    
    data_2019['knn_label'] = np.where(pred_k_19 == 1, 'Green','Red')
    
    # trade with knn predicted labels
    # function imported from helper package
    knn_trading_19 = trading_strategy(data_2019, data_2019['knn_label'])
    mean_pred_trading_19 = np.array(list(knn_trading_19.values())).mean().round(2)
    
    # buy and hold strategy
    buy_n_hold_19 = buy_and_hold(data_2019)
    
    print('\nConclusion: ')
    print('\nWith predicted labels for 2019:\n'
          f'Average of portfolio balance: ${mean_pred_trading_19}')
    
    print('\nAt the end of the year, trading with labels predicted by KNN is more'
          ' profitable as compared to buy and hold strategy.'
          f'\nKNN startegy resulted in an amount of ${knn_trading_19[52]}'
          f' while buy and hold resulted in ${buy_n_hold_19[52]}.')
    
    
          
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)       