"""
Neha Bais
MET CS-677 Assignment-6.1b

Implementing KNN & Logistic classifier using SHAPLEY

1. Compute marginal cntributions selecting a single feature set 
   for IRIS dataset using  logistic regression.
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


def calculating_accuracy(features, labels, data):
    '''
    Calculating accuracy selecting 
    all features value.
    '''
    X = data[features].values
    le = LabelEncoder()
    Y = le.fit_transform(labels)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5,
                                                         random_state = 3)
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(X_train, Y_train)
    prediction =  log_reg_classifier.predict(X_test)
    accuracy = round(np.mean(prediction == Y_test)* 100 , 2)
        
    return accuracy

    
try :
    url = r'https://archive.ics.uci.edu/ml/' + \
        r'machine-learning-databases/iris/iris.data'
    
    data = pd. read_csv (url , names =[ 'sepal-length', 'sepal-width',
                                           'petal-length', 'petal-width', 'Class'])
    
    features = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
    class_labels = ['Iris-versicolor', 'Iris-virginica', 'Iris-setosa']
    
    
    ####################
    # Removing one feature value at a time
    ####################
    
    # removing sepal-length as the feature value
    feature_s_length = [x for i, x in enumerate(features) if i != 0]
    # removing sepal-width as the feature value
    feature_s_width = [x for i, x in enumerate(features) if i != 1]
    # removing petal-length as the feature value
    feature_p_length = [x for i, x in enumerate(features) if i != 2]
    # removing petal-width as the feature value
    feature_p_width = [x for i, x in enumerate(features) if i != 3]
    
    
        
    ################
    #  Versicolor  #
    ################
    versicolor = data.loc[:,'Class'].apply(lambda x: 1 
                                                   if x == 'Iris-versicolor' else 0)
    
    # all features
    acc_versicolor_all = calculating_accuracy(features, versicolor.values, data)
    
    # accuracy when sepal-length is removed from feature
    acc_vers_seplen = calculating_accuracy(feature_s_length, versicolor.values, 
                                            data)
    
    # accuracy when sepal-width is removed from feature 
    acc_vers_sepwidth = calculating_accuracy(feature_s_width, versicolor.values, 
                                           data)
    
    # accuracy when petal-length is removed from feature
    acc_vers_petlen = calculating_accuracy(feature_p_length, versicolor.values, 
                                           data)
    
    # accuracy when petal-width is removed from feature
    acc_vers_petwidth = calculating_accuracy(feature_p_width, versicolor.values, 
                                           data)
    
    
    ################
    #  Virginica   #
    ################
    virginica = data.loc[:,'Class'].apply(lambda x: 1 
                                                   if x == 'Iris-virginica' else 0)
    # all features
    acc_virginica_all = calculating_accuracy(features, virginica.values, data)
    
    # accuracy when removing sepal-length as the feature value
    acc_virg_seplen = calculating_accuracy(feature_s_length, virginica.values, 
                                            data)
    
    # accuracy when removing sepal-width as the feature value
    acc_virg_sepwidth = calculating_accuracy(feature_s_width, virginica.values, 
                                           data)
    
    # accuracy when removing petal-length as the feature value
    acc_virg_petlen = calculating_accuracy(feature_p_length, virginica.values, 
                                           data)
    
    #  accuracy when removing petal-width as the feature value
    acc_virg_petwidth = calculating_accuracy(feature_p_width, virginica.values, 
                                           data)
    
    ################
    #    Setosa    #
    ################
    setosa = data.loc[:,'Class'].apply(lambda x: 1 
                                                   if x == 'Iris-setosa' else 0)
    # all features
    acc_setosa_all = calculating_accuracy(features, setosa.values, data)
    
    # accuracy when removing sepal-length as the feature value
    acc_set_seplen = calculating_accuracy(feature_s_length, setosa.values, 
                                            data)
    
    # accuracy when removing sepal-width as the feature value
    acc_set_sepwidth = calculating_accuracy(feature_s_width, setosa.values, 
                                           data)
    
    # accuracy when removing petal-length as the feature value
    acc_set_petlen = calculating_accuracy(feature_p_length, setosa.values, 
                                           data)
    
    # accuracy when removing petal-width as the feature value
    acc_set_petwidth = calculating_accuracy(feature_p_width, setosa.values, 
                                           data)
     
    ##############################
    #    Marginal Contributions  #
    ##############################
    marginal_vers_seplen = round(acc_versicolor_all - acc_vers_seplen, 2)
    marginal_vers_sepwid = round(acc_versicolor_all - acc_vers_sepwidth, 2)
    marginal_vers_petlen = round(acc_versicolor_all - acc_vers_petlen, 2)
    marginal_vers_petwid = round(acc_versicolor_all - acc_vers_petwidth, 2)
    
    marginal_virg_seplen = round(acc_virginica_all - acc_virg_seplen, 2)
    marginal_virg_sepwid = round(acc_virginica_all - acc_virg_sepwidth, 2)
    marginal_virg_petlen = round(acc_virginica_all - acc_virg_petlen, 2)
    marginal_virg_petwid = round(acc_virginica_all - acc_virg_petwidth, 2)
    
    marginal_set_seplen = round(acc_setosa_all - acc_set_seplen, 2)
    marginal_set_sepwid = round(acc_setosa_all - acc_set_sepwidth, 2)
    marginal_set_petlen = round(acc_setosa_all - acc_set_petlen, 2)
    marginal_set_petwid = round(acc_setosa_all - acc_set_petwidth, 2)
    
    # Printing output to console
    marginal_data = {
             'Versicolor': [marginal_vers_seplen, marginal_vers_sepwid, 
                               marginal_vers_petlen, marginal_vers_petwid],
             'Virginica':  [marginal_virg_seplen, marginal_virg_sepwid, 
                               marginal_virg_petlen, marginal_virg_petwid],
              'Setosa':    [marginal_set_seplen, marginal_set_sepwid, 
                               marginal_set_petlen, marginal_set_petwid] }
             
    final_data_frame = pd.DataFrame(marginal_data, 
                                    columns = ['Versicolor','Virginica','Setosa'],
                                    index = ['Sepal length', 'Sepal Width', 
                                             'Petal length', 'Petal Width'])
    
    print('\t\tMarginal Contributon for IRIS dataset ')
    print('\t\t-------------------------------------')
    print(final_data_frame)
    
    print('\nConclusion')
    print('1. Versicolor --> removing Sepal width as the feature value results in '
          'reduction of accuracy at a larger extent. Sepal width is an important feature for Versicolor.\n'
          '2. Virginica --> When sepal length is removed as feature value, accuracy remains same.\n'
          'While when any of the other 3 features are removed, accuracy increases. So, the other ' 
          ' three features seems not so important in case of Virginica, removing any of the 3 '
          ' and keeping Sepal length each time increases the accuracy. Hence, Sepal length is an '
          'important feature.\n'
          '3. Setosa --> removing any of the feature value, does not impact the accuracy. '
          'Accuracy remains the same in all cases. So we can say removing any of the '
          'feature value does not matter in this case.')
    
except Exception as e:
    print(e)
    print('Failed to read iris data')      

