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
from sklearn.linear_model import LinearRegression



ticker = 'BAC'
input_dir = os.path.dirname(os.path.realpath('__file__'))
ticker_file = os.path.join(input_dir, ticker + '.csv')


def plot_figure(x, y, title, xlabel, ylabel) :
    plt.figure( figsize =(10 ,4))
    ax = plt.gca()
    ax. xaxis.set_major_locator(MaxNLocator( integer = True ))
    plt.plot(x, y, color ='red', linestyle ='dashed',
                marker ='o', markerfacecolor ='black', markersize = 3)
    plt.title (title)
    plt.xlabel (xlabel)
    plt.ylabel (ylabel)  
    plt.show()
    

def lin_reg_trading(X, Y, w) :
    profit_loss_list = []
    position = 'None'
    
    short_shares = 0.00
    long_shares  = 0.00
    
    short_amount = 0.00
    long_amount  = 0.00
    
    long_count = 0
    short_count = 0
    
    long_profit = 0.00
    short_profit = 0.00
       
    for i in range(len(X)-w) :
               
        lin_reg = LinearRegression(fit_intercept = True)
        lin_reg.fit(X[i : i+w], Y[i : i+w])
        predicted = lin_reg.predict(X[i+w].reshape(-1,1))
        
        if predicted > Y[i+w-1] :
            # no position
            if position == 'None' :
                long_shares = 100.00 / Y[i+w-1]
                long_amount = Y[i+w-1]
                position = 'long'
                long_count += 1
                            
            elif position == 'short' :
                 profit_loss = short_shares * (short_amount - Y[i+w-1])
                 profit_loss_list.append(profit_loss)
                 short_profit += profit_loss
                 short_amount = 0.00
                 position = 'None'
        
            # do nothing when position == 'long'
    
        elif  predicted < Y[i+w-1] :
            # no position
            if position == 'None' :
               short_shares = 100.00 / Y[i+w-1]
               short_amount = Y[i+w-1]
               position = 'short'
               short_count += 1
               
            elif position == 'long':
                profit_loss = long_shares * (Y[i+w-1] - long_amount)
                profit_loss_list.append(profit_loss)
                long_profit += profit_loss
                long_amount = 0.00
                position = 'None'
                
            # do nothing when position == 'short'    
                
    return profit_loss_list , long_count, short_count, long_profit, short_profit


try :
    df = pd.read_csv(ticker_file)
        
    # extract 2018 and 2019 data separately
    data_2018 = df.loc[df['Year'] == 2018].copy()
    data_2019 = df.loc[df['Year'] == 2019].copy()
    
    day_2018 = np.arange(0,len(data_2018['Adj Close']))
    adj_close_18 = data_2018['Adj Close'].values
    
    X_2018 = day_2018.reshape(len(day_2018),1)
    Y_2018 = adj_close_18.reshape(len(adj_close_18),1)
    
    window = np.arange(5,31)
    avg_p_l_18 = {}
    
    for w in window :
        p_l_18, l_count_18, s_count_18, l_profit_18, s_profit_18 =    \
                                            lin_reg_trading(X_2018, Y_2018, w)
        avg_p_l_18[w] = round(np.mean(p_l_18) , 2)
    
    
    # Plotting avergae profit loss for different window
    plot_figure(window, np.array(list(avg_p_l_18.values())), 
                'Average Profit/Loss vs. w for 2018 BAC stock Subset ', 
                'Window Size : w', 'Profit/Loss')
        
    # optimal value for w --> 27 with a profit of 0.46
    optimal_w = max(avg_p_l_18, key=lambda key: avg_p_l_18[key])
    print('\nQuestion -1 ')
    print(f'Optimal value of w for 2018 = {optimal_w}')
    
    
    #####################################################
    #                       2019                        #
    #####################################################

    ####### Question - 2            

    adj_close_19 = data_2019['Adj Close'].values
    day_2019 = np.arange(0,len(data_2019['Adj Close']))
    
    
    X_2019 = day_2019.reshape(-1,1)
    Y_2019 = adj_close_19.reshape(-1,1)
    
    predicted_19 = {}
    r_score = {}
    
    for i in range(len(X_2019)-optimal_w) :
        model = LinearRegression()
        model.fit(X_2019[i : i+optimal_w], Y_2019[i : i+optimal_w])
        predicted_19[i] = model.predict(X_2019[i+optimal_w].reshape(-1,1))
                
        r_score[i] = round( model.score(X_2019[i : i+optimal_w], 
                                        Y_2019[i : i+optimal_w]) , 2)

    # plot for r2_score
    plot_figure(day_2019[optimal_w : ], np.array(list(r_score.values())), 
                'r2_score for 2019 BAC stock Subset ', 'Days', 'r2 score')
    
    
    # average r**2 value
    avg_r2_score = round( sum(r_score.values()) / len(r_score) , 2)
    print('\nQuestion -2 ')
    print(f'Average r**2 = {avg_r2_score}')
    print('\nFrom the above graph we can see that for around half of the year the r score value '
          'is less than 50% and for the other half it is greater than 50%. '
          'Also, I observed that among 225 days, for only 6 days r score( > 0.90) is somewhat '
          'close to 1 and for around 43 days, it is closer to 0. '
          'So, in this case, we cant say whether linear regression model fits well '
          'for our data as half of the time it is performing I can say good and for the remaining '
          'half it is not fitting the data properly. ')
    
    
    ############################################################
    #         Trading for 2019 with optimal_w from 2018        #
    ############################################################
    
    X_2019 = day_2019.reshape(len(day_2019),1)
    Y_2019 = adj_close_19.reshape(len(adj_close_19),1)
    
    p_l_19, l_count_19, s_count_19, l_profit_19, s_profit_19 =  \
                                    lin_reg_trading(X_2019, Y_2019, optimal_w)
    avg_p_l_19 = round(np.mean(p_l_19) , 2) 
        
   
    ##### Question - 3
    print('\nQuestion -3 ')
    print(f'Using optimal w as {optimal_w} from 2018 -')
    print(f'Number of long position transactions for 2019 -> {l_count_19}')
    print(f'Number of short position transactions for 2019 -> {s_count_19}')
      
    ##### Question - 4
    avg_long_pl_19 = round(l_profit_19[0] / l_count_19, 2 )
    avg_short_pl_19 = round(s_profit_19[0] / s_count_19, 2 )
    
    print('\nQuestion -4 ')
    print(f'Average profit/loss per long position trade for 2019 -> ${avg_long_pl_19}')
    print(f'Average profit/loss per short position trade for 2019 -> ${avg_short_pl_19}')
    
    
    ########## Question = 5
    avg_long_short_days_19 = round( (l_count_19 + s_count_19) / 2)
    print('\nQuestion -5 ')
    print(f'Average days for long & short positions for 2019 -> {avg_long_short_days_19}')
    
    
    ########## Question = 6
    # 2018 calculation using w* 
    
    p_l_w_18, l_count_w_18, s_count_w_18, l_profit_w_18, s_profit_w_18 =  \
                                    lin_reg_trading(X_2018, Y_2018, optimal_w)
                                    
    avg_long_pl_18 = round(l_profit_w_18[0] / l_count_w_18, 2 )
    avg_short_pl_18 = round(s_profit_w_18[0] / s_count_w_18, 2 )
    
    avg_long_short_days_18 = round( (l_count_w_18 + s_count_w_18) / 2)                                
    
    trading_data = {
        '2018' : [avg_p_l_18[27], l_count_18, s_count_18, 
                  avg_long_pl_18, avg_short_pl_18, avg_long_short_days_18 ],
        '2019' : [avg_p_l_19, l_count_19, s_count_19, 
                  avg_long_pl_19, avg_short_pl_19, avg_long_short_days_19 ]}
    
    trading_df = pd.DataFrame(trading_data,
                              columns = ['2018', '2019'],
                              index = ['Average P/L', 'Nbr of Long Position', 
                                       'Nbr of Short Position', 'Avg Long P/L' ,
                                       'Avg Short P/L' , 'Avg long/Short days'])
    
    print('\nQuestion -5   For w = 27')
    print('---------------------------')
    print(trading_df)
    print('\n 1. As per the above results, the results are not same for both the years.\n '
          '2. For 2018 using the optimal w we experienced a profit, while for 2019, loss occured.\n '
          '3. For 2018, Long transactions resulted in loss, while short position was profitable.\n'
          ' 4. This was not the case for 2019, as in 2019, long position was profitable but short position resulted in loss.')
    
except Exception as e:
    print(e)
    print('Failed to read stock data for ticker', ticker)    