"""
Neha Bais
MET CS-677 Assignment-10

Implementing SVM classifier for below 3 kerels to predict labels for year 2019
- Linear
- Gaussian
- Polynomial (degree = 2)

1. Compute accuracy for all three SVMs and compare them. Which one is better?
2. Compute confusion matrix and true +ve and -ve rate for LSVM
3. Implement trading strategy with predicted LSVM labels
   and compare the results with buy and hold strategy. 
   Which strategy is more profitable ?
"""

import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn import svm 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from helper_function import *

def evaluate_model(x_train, x_test, y_train, y_test, k):
    '''
    Predict labels for test dataset training on train dataset
    Returns Accuracy
    '''
    svm_clf = svm.SVC(kernel = k)
    svm_clf.fit(x_train, y_train)
    prediction = svm_clf.predict(x_test)
    accuracy = round(np.mean(prediction == y_test) * 100 , 2)
    return accuracy, prediction
    
    
# Load the Dataset
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
        
    # feature scaling
    scaler = StandardScaler()
    scaler.fit(X_18_features)
    X_train_scaled = scaler.transform(X_18_features)
    X_test_scaled = scaler.transform(X_19_features)
    
    #########################################
    #   Linear Support Vector Classifier    #
    #########################################
    linear_accuracy, l_pred = evaluate_model(X_train_scaled, X_test_scaled, 
                                     Y_18_class, Y_19_class, 'linear')
    
    print('\n\t\t SVM classifier')
    print('\t   ---------------------------\n')
    print(f'1. Accuracy for Linear SVM : {linear_accuracy}%')
        
    # Confusion Matrix
    linear_svm_cf = confusion_matrix(Y_19_class, l_pred)
    print(f'2. Confusion Matrix for Linear SVM : \n{linear_svm_cf}')
    #sns.heatmap(linear_svm_cf, annot= True, cmap = 'Blues',
    #           xticklabels = ['Green', 'Red'], 
    #            yticklabels = ['Green', 'Red'])
    
    # True +ve and -ve rate for 2019
    tn, fp, fn, tp = linear_svm_cf.ravel()
    
    true_pos_rate = tp / (tp + fn)
    true_neg_rate = tn / (tn + fp)
    
    print(f'True -ve  = {tn}  False +ve = {fp} \n'
          f'False -ve = {fn}   True +ve  = {tp}\n')
    
    print(f'3(a). True positive rate for 2019 = {true_pos_rate*100:.2f}%.')
    print(f'3(b). True negative rate for 2019 = {true_neg_rate*100:.2f}%.')
    
    
    #########################################
    #   Gausian Support Vector Classifier   #
    #########################################
    gausian_accuracy, g_pred = evaluate_model(X_train_scaled, X_test_scaled, 
                                     Y_18_class, Y_19_class, 'rbf')
    
    print(f'\n4. Accuracy for Gausian SVM : {gausian_accuracy}%')
    
    
    ############################################
    #   Polynomial Support Vector Classifier   #
    ############################################
    svm_classifier = svm.SVC(kernel = 'poly', degree = 2)
    svm_classifier.fit(X_train_scaled, Y_18_class)    
    poly_pred = svm_classifier.predict( X_test_scaled )
    poly_accuracy = round(np.mean(poly_pred == Y_19_class) *100 , 2)
    
        
    print(f'5. Accuracy for Polynomial SVM (degree-2): {poly_accuracy}% \n')
    
    # Comparision of different SVM classifiers
    acc_data = {
             'Linear SVM': linear_accuracy,
             'Gausian SVM': gausian_accuracy,
             'Polynomial SVM': poly_accuracy }
    
    final_data_frame = pd.DataFrame(acc_data, index = ['Accuracy'])
    print('Comparision of Accuracy of different SVMs')
    print('-----------------------------------------')
    print(final_data_frame)   
    
    print('\nGaussian SVM gives a better performance among the 3 different kernels.'
          'Polynomial SVM prvides the least accuracy compared to Linear& Gaussian.')   

                             
    ##############################################################
    #              Trading for 2019 with predicted labesl        #
    ##############################################################
    
    data_2019['LSVM_labels']  = np.where(l_pred == 1, 'Red','Green')
    
    LSVM_trading = trading_strategy(data_2019, data_2019['LSVM_labels'])
    buy_n_hold = buy_and_hold(data_2019)
    
    profitable_strategy = 'Linear SVM' if LSVM_trading[len(LSVM_trading)-1] > \
            buy_n_hold[len(buy_n_hold)-1] else 'Buy & hold'    
    
       
    # Plot portfolio balance for 2 strategies
    plt.figure( figsize =(10 ,4))
    ax = plt. gca ()
    ax. xaxis.set_major_locator(MaxNLocator( integer = True ))
    plt.plot(np.array(list(LSVM_trading.keys())), np.array(list(LSVM_trading.values())), 
                color ='red', linestyle ='solid',label = 'Linear SVM')
    
    plt.plot(np.array(list(buy_n_hold.keys())), np.array(list(buy_n_hold.values())), 
                color ='green', linestyle ='solid',label = 'Buy & Hold')
                                
    plt.legend()
    plt.title ('2019 - Portfolio balance for 2 trading strategies')
    plt.xlabel ('Week Numbers')
    plt.ylabel ('Portfolio Balance')  
    plt.show()
        
    print('\nConclusion :')
    print(f'At end of year {profitable_strategy} resulted in a larger amount. \n'
          f'Portfolio balance for Linear SVM at EOY is ${LSVM_trading[len(LSVM_trading)-1]} \n'
          f'Portfolio balance for buy & hold at EOY is ${buy_n_hold[len(buy_n_hold)-1]}')
    
    
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)    