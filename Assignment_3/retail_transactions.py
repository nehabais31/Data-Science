"""
Created on Sun Jun  7 13:20:25 2020

Neha Bais
MET CS-677 Assignment - 3.2 (Retail data analysis) 

1. Histograms for the frequencies for real distribution,equal-weight and Bernford (for each digit)

2. Histograms for the relative errors for equal wt and benford model (for each digit)

3. RMSE for above 2 models

4. freq distribution analysis for 3 countries: Italy, Israel, Singapore

"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


input_dir = os.path.dirname(os.path.realpath('__file__'))
input_file = os.path.join(input_dir, 'online_retail.csv')


def absolute_error(actual, predicted) :
    return ( abs(actual - predicted) / actual ) 
       

def rmse(actual, predicted) :
    '''
    returns mean square root error
    '''
    rms_err = np.sqrt(np.mean((actual - predicted)**2))
    return round(rms_err,2)


def country_leading_digit(df, country):
    '''
    Calculates the leading digit based on country name
    Returns a dict with the count of each leading digit 
    '''
    country_price = df.loc[df['Country'] == country, 'Price'].abs()
    
    # Extract first digit and stor its count in dictionary
    country_lead_digit = country_price.apply(lambda x: str(x)[0] if x >= 1 else None)
    country_digit_count = country_lead_digit.value_counts().to_dict()
    
    country_digit_count = {int(k): int(v) for k, v in country_digit_count.items()}
    
    # RElative freq of leading digit
    real_dist_country = {}
    digit_list = [i for i in range(1,10)]
    
    for key in digit_list :
        if key in country_digit_count:
            real_dist_country[key] = round( country_digit_count[key] / sum(country_digit_count.values()) ,4)
        else :
            real_dist_country[key] = 0.0
    
    real_dist_values = list( real_dist_country.values() )
    
    return real_dist_values


def plot_histogram(freq, heading, y_naming) :
    '''
    Function to plot histogram for specified model
    '''
    x_pos = np.arange(1,10)

    plt.title("Histogram of {}".format(heading))
    plt.bar(x_pos, freq, width = 0.5, align = 'center', color='c', edgecolor='red', label = heading )
    
    plt.xticks(x_pos)
    plt.xlabel('Digit', fontweight = 'bold')
    plt.ylabel(y_naming, fontweight = 'bold')
    plt.legend()
    plt.show()  
    

try :
    #df = pd.read_csv(input_file)
    df = pd.read_csv(r'E:\Neha\Course_Materials\Summer2020\Data_Science\Assignments\Assignment_3\online_retail.csv')
    
    # Calculating Frequencies for Benford's law Model
    digits = [ i for i in range(1,10) ]
    benford_freq = []
    
    for d in digits :
        benford_freq.append(np.log10(1 + 1/d).round(3))

    # Calculating equal weight distribution freq
    equal_wt_freq = [round((1/9) , 3)] * 9
     
    # Calculating frequencies of digits for Price column of our data
    # Getting the absolute values of price
    price = df['Price'].abs()
    
    # Extracting the first digit excluding 0 as first digit
    lead_digit = price.apply(lambda x: str(x)[0] if x >= 1 else None)

    # Getting the total counts of each digit     
    lead_digit_count = lead_digit.value_counts()
    lead_digit_count = lead_digit_count.to_dict()
    
    lead_digit_count = {int(k): int(v) for k, v in lead_digit_count.items()}
    
    # RElative frequncies of leading digit across Price column of dataset
    real_distribution = {}
    for key in digits:
        if key in lead_digit_count :
            real_distribution[key] = round( lead_digit_count[key] / sum(lead_digit_count.values()) ,4)
        else :
            real_distribution[key] =  0.0
    
    real_dist_values = list( real_distribution.values() )
    
    '''
    Question-1: Histogram for all 3 models
    '''
    # Plotting the Histogram for all 3 Models
    plot_histogram(benford_freq ,'Benford freq', 'Frequency')
    plot_histogram(equal_wt_freq, 'Equal weight freq', 'Frequency')
    plot_histogram(real_distribution.values(), 'real distribution', 'Frequency')
    
    '''
    Question-2: Histogram for relative errors for 2 Models
    '''
    # Calculating the relative errors
    err_metric_eq_wt     = {}
    err_metric_benford  = {}
    
    for d in digits :
        err_metric_eq_wt[d]    = absolute_error(real_dist_values[d-1] , equal_wt_freq[d-1])
        err_metric_benford[d] = absolute_error(real_dist_values[d-1] , benford_freq[d-1])
    
    # Plotting histogram 
    print('\nQuestion-2: Histograms for Absolute errors')
    plot_histogram(err_metric_eq_wt.values(), 'Absolute error: Equal weight', 'Absolute error')  
    plot_histogram(err_metric_benford.values(), 'Absolute error: Benford model', 'Absolute error')  
    
    print('\nConclusion: \nObserving the patterns for absolute error for the 2 models,'
          ' it is concluded that the relative errors is comparatively smaller in Benford model for most of the digits.')
    
    '''
    Question-3: RMSE for 2 models
    '''
    # For model-1 (equal weight distribution)
    rmse_model_1 = rmse(np.array(real_dist_values), np.array(equal_wt_freq))
      
    # For model 2 ( Bernford law distribution)
    rmse_model_2 =  rmse(np.array(real_dist_values), np.array(benford_freq))
   
    print('\nQuestion-3: ')
    print('RMSE for Equal weighted model: ', rmse_model_1)
    print('RMSE for Benford Model: ', rmse_model_2)
    print('Conclusion: ', end = ' ')
    print('Equal distribution model is closer to real distribution.'
          if rmse_model_1 < rmse_model_2 else 
          'Benford Model is closer to real distribution.')
     
    
    '''
    Question-4 : Leading digit distribution for 3 countries
    Italy, Singapore, Israel
    '''
    
    leading_digit_Italy = country_leading_digit(df, 'Italy')
    leading_digit_Singapore = country_leading_digit(df, 'Singapore')
    leading_digit_Israel = country_leading_digit(df, 'Israel')
    
    # RMSE calculation for these 3 countries
    rmse_italy     =  rmse(np.array(leading_digit_Italy) , np.array(equal_wt_freq))
    rmse_singapore =  rmse(np.array(leading_digit_Singapore) , np.array(equal_wt_freq))
    rmse_israel    =  rmse(np.array(leading_digit_Israel) , np.array(equal_wt_freq))
    
    # RMSE calculation for these 3 countries
    rmse_italy_bn     =  rmse(np.array(leading_digit_Italy) , np.array(benford_freq))
    rmse_singapore_bn =  rmse(np.array(leading_digit_Singapore) , np.array(benford_freq))
    rmse_israel_bn    =  rmse(np.array(leading_digit_Israel) , np.array(benford_freq))
    
    print('\nrmse_italy_bn' , rmse_italy_bn,\
          'rmse_singapore_bn' , rmse_singapore_bn,\
          'rmse_israel_bn', rmse_israel_bn)
        
    print('\rmse_italy_eq_wt' , rmse_italy,\
          'rmse_singapore_eq_wt' , rmse_singapore,\
          'rmse_israel_eq_wt', rmse_israel)   
        
    
    print('\nQuestion-4 :')
    print('Out of Singapore, Italy and Israel: ')
    if rmse_italy < rmse_singapore  and  rmse_italy < rmse_israel :
        print('For Italy, distribution is closest to equal weight.')
    elif rmse_singapore < rmse_italy  and rmse_singapore < rmse_israel :
        print('For Singapore, distribution is closest to equal weight.')
    else :
        print('For Israel, distribution is closes to equal weight.')
    
    print('\nQuestion-5 :')
    print('Conclusion: From this retail data analysis, it is concluded that '
          ' our data follows the Benford model. The most frequent occuring digit is 1 and 2.'
          ' The relative errors for these digits is also significantly smaller as compared to other digits.')
    
except Exception as e:
    print(e)
    print('Unable to read input file.')            
    