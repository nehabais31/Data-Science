"""
Created on Sat May 23 12:30:31 2020

@author: Neha Bais
this scripts reads your ticker file (e.g. BAC.csv) and
constructs a list of lines

Day trading strategy for BAC - (2014 - 2019)
"""
import os
import matplotlib.pyplot as plt

def threshold_strategy(stock_open, stock_close) :
    invest_amount = 100
    long_profit = []
    sell_short_profit = []
    
    avg_daily_profit = {}
    
    profit_l = 0
    profit_s = 0
    
    count_l = count_s = 0
    
    
    # set a threshold from 0-10% by dividing it between 100 points 
    threshold = [i / 1000 for i in range(101)]
    
    for limit in range(len(threshold)) :
        # for these 100 limits, trade only if overnight return > threshold_limit
        for i in range(1, len(stock_open)):
            if (stock_open[i] > stock_close[i-1]) and ((abs(stock_open[i] - stock_close[i-1]) / stock_close[i-1]) > threshold[limit]) :  # buy
                    shares = invest_amount / stock_open[i]
                    profit_l += shares * (stock_open[i] - stock_close[i])  
                    count_l += 1
             
            elif (stock_open[i] < stock_close[i-1])  and ((abs(stock_open[i] - stock_close[i-1]) / stock_close[i-1]) > threshold[limit]) :     # sell_short
                shares = invest_amount / stock_open[i]
                profit_s += shares * (stock_close[i] - stock_open[i]) 
                count_s += 1
                
            # reset shares and cost_eod at end of a day
            # since each day we are investing $100 as per market trend    
            shares = 0    
        
        # store sum of profits for each threshold in 2 separate lists
        # each for long and short positions
        if count_l != 0 :
            avg_long_profit =  profit_l / count_l
        else :
            avg_long_profit = 0
            
        # store long profit in a list for each threshold limit    
        long_profit.append(round(avg_long_profit,2))   
        
        if count_s != 0 :
            avg_short_profit = profit_s / count_s
        else :
            avg_short_profit = 0
            
        # store short sell profit in a list for each threshold limit       
        sell_short_profit.append(round(avg_short_profit,2))
        
        # calculate average profit at each threshold 
        avg_profit = round((profit_l + profit_s) / len(stock_open) , 2)
        
        # storing avergae profit per trade according to the threshold limit
        if threshold[limit] not in avg_daily_profit: 
            avg_daily_profit[threshold[limit]] = avg_profit

                
        # reset profit and count values for each turn
        profit_l = profit_s = count_l = count_s = 0            
    
    print('\nAverage profit per threshold: \n',avg_daily_profit)
    
    dict_values = sorted(avg_daily_profit.items())        # sorted by key, return a list of tuples
    x, y = zip(*dict_values)                              # unpack a list of pairs into two tuples
    
    plt.plot(x , y)
    plt.xlabel('Threshold limits')
    plt.ylabel('Average profit')
    plt.title('BAC')
    plt.show()
    
    print('\nConclusion: \nThe optimal threshold limit is 0 - 0.5% where the profit is maximum: $5.86 average per trade.'
          ' But, after 0.5% the profit continues to fall.')
      
    
    # long position analysis
    print('\nLong position analysis for each threshold: ')
    plt.plot(threshold, long_profit)
    plt.xlabel('Threshold limits')
    plt.ylabel('Long position')
    plt.show()
        
    #print('\nlong_profit: \n', long_profit)
    
    # short position analysis
    print('\nShort sell analysis for each threshold: ')
    plt.plot(threshold, sell_short_profit)
    plt.xlabel('Threshold limits')
    plt.ylabel('Short position')
    plt.show()
    
    #print('\nsell_short_profit: \n', sell_short_profit)
    
    print('\nAs per the data for each threshold - \nfor long position, '
          'we found that there is a rise in average profit at each threshold limit. '
          'But in case of short sell, upto the threshold limit 0 - 1.1% we are short selling '
          'but experienced a loss, and after that we are not short selling anymore for the rest of the threshold points. ')
        
    
def main():
    ticker = 'BAC'
    input_dir = os.path.dirname(os.path.realpath('__file__'))
    ticker_file = os.path.join(input_dir, ticker + '.csv')

    try:   
        with open(ticker_file) as f:
            lines = f.read().splitlines()
        print('opened file for ticker: ', ticker)
    
        line_values = []               #list to hold row values
        
        # Store each row into list; each list separated by comma
        for line in lines:
            line_values.append(line.split(','))
        
        stock_open = []     # list to store open market values
        stock_close = []    # list to store adj_close values
        
        # creating lists of open and adj close values
        for i in range(1, len(line_values)):
            stock_open.append(float(line_values[i][7]))
            stock_close.append(float(line_values[i][12]))
        
        '''
        do nothing at first day of trading
        at second day - decide long or short position
        if open(current_day) > close(previous_day) --> LONG
        if open(current_day) < close(previous_day) --> SHORT
        daily invest amount = $100
        '''    
            
        sell_short_profit = []    
        long_profit  = [] 
        invest_amount = 100
        
        for i in range(1, len(stock_open)):     # exclude header
            if stock_open[i] > stock_close[i-1] :        # buy
                shares = invest_amount / stock_open[i]
                long_profit.append( shares * (stock_open[i] - stock_close[i]) ) 
                                        
            elif stock_open[i] < stock_close[i-1]  :     # sell_short
                shares = invest_amount / stock_open[i]
                sell_short_profit.append(shares * (stock_close[i] - stock_open[i]) )
                
            # reset shares and cost_eod at end of a day
            # since each day we are investing $100 as per market trend    
            shares = 0

        
        # calculate sell_short & long average profit
        avg_sell_short = sum(sell_short_profit) / len(sell_short_profit)
        avg_long_profit = sum(long_profit) / len(long_profit)
    
        # Q-1: What is the average daily profit  ?
        avg_daily_profit = ( sum(sell_short_profit) + sum(long_profit) ) / ( len(sell_short_profit) + len(long_profit) )
        print('\nAverage daily profit: ' , round(avg_daily_profit, 2) )
    
    
        # Q-2: What is more profitable: long or short positions? 
        print('\nAverage short profit:' , round(avg_sell_short, 2) )
        print('Average long profit: ' , round(avg_long_profit, 2))
    
        if avg_sell_short > avg_long_profit :
            print('In my case, the short sell is more profitable as compared to the long position.')
        elif avg_sell_short < avg_long_profit:
            print('In my case, the long position is more profitable as compared to the short sell.')
        else:
            print('Both short sell and long position are giving equal amount of profits.')
    
        # Q-3: Compute avg profit per trade by setting a restriction of trading 
        threshold_strategy(stock_open, stock_close)   
          
            
    except Exception as e:
        print(e)
        print('failed to read stock data for ticker: ', ticker)

main()