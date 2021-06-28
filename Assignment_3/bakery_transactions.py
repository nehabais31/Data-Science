"""
Created on Sat Jun  6 12:06:08 2020

Neha Bais
MET CS677 - Assignment - 3.1

Analysing Bakery Transactions

"""

import pandas as pd
import numpy as np
import os


input_dir = os.path.dirname(os.path.realpath('__file__'))
input_file = os.path.join(input_dir, 'Assignment_3', 'BreadBasket_DMS_output.csv')

try :
    df = pd.read_csv(input_file)
    
    #--------Ques-1-----------------
    print('\n********** Question 1 ********** ')
    
    # Busiest hour in terms of transactions
    busy_hour = df.groupby('Hour')['Transaction'].nunique().nlargest(1)
    print('a) 11 is the busiest hour with 1445 transactions.')
    
    # Busiest day in terms of transactions
    busy_day_of_week = df.groupby('Weekday')['Transaction'].nunique().nlargest(1)
    print('\nb) Saturday is the busiest day of the week with 2068 transactions.')
    
    # Busiest period in terms of transactions
    busy_period = df.groupby('Period')['Transaction'].nunique().nlargest(1)
    print('\nc) Afternoon is the busy period with 5307 transactions.')
    
    
    #----------- Ques-2 -----------------------
    print('\n********** Question 2 ********** ')
    
    # Most profitable hour in terms of revenue
    profitable_hour = df.groupby('Hour')['Item_Price'].sum().nlargest(1)
    print('a) 11 is the most profitable hour with transactions of worth $21453.44.')
    
    # Most profitable day of week
    profitable_day = df.groupby('Weekday')['Item_Price'].sum().nlargest()
    print('\nb) Saturday is the most profitable day with transactions of worth $31531.83.')
    
    # Most profitable period
    profitable_period = df.groupby('Period')['Item_Price'].sum().nlargest()
    print('\nc) Afternoon is the most profitable period with transactions of worth $81299.97.')
 
    
    #------------- Ques-3 ------------------------
    print('\n********** Question 3 ********** ')
    
    # Most popular item
    most_pop_item = df['Item'].value_counts().nlargest()                                
    print('a) Coffee is the most popular item.')

    # Least popular items
    least_pop_item = df['Item'].value_counts().nsmallest(10)
    print('b) Least popular items are: \n   Gift voucher \n   Olum & polenta \
          \n   Bacon \n   Adjustment \n   Chicken sand \n   The BART \
              \n   Polenta \n   Raw bars')                             
     

    #------------------ Ques-4 ------------------------------
    print('\n********** Question 4 ********** ')     
          
    # Calculating total unique transactions per day of the week groupby year, month, day and weekday
    weekday_trans = df.groupby(['Year','Month','Day','Weekday'])\
                                 ['Transaction'].nunique().to_frame()
            
    # Calculating maximum transactions per day of the week
    max_weekday_trans = weekday_trans.groupby('Weekday')['Transaction'].max().to_frame()
        
    # Finding number of barristas required for each day of week
    # Given Each barrista can handle 50 transactions per day    
    barristas = (np.ceil(max_weekday_trans.Transaction / 50)).astype(int)
    
    print('Number of barristas needed for each day of week:\n', barristas)
    

    #------------------- Ques -5 ------------------------------
    print('\n********** Question 5 ********** ') 
    
                                
    food = ['Bacon', 'Baguette', 'Bakewell', 'Bare Popcorn', 'Basket',
		'Bowl Nic Pitt', 'Bread', 'Bread Pudding', 'Brioche and salami',
		'Brownie', 'Cake', 'Caramel bites', 'Cherry me Dried fruit',
		'Chicken Stew', 'Chicken sand', 'Chocolates', 'Christmas common',
		'Cookies', 'Crepes', 'Crisps', 'Duck egg', 'Dulce de Leche',
		'Eggs', "Ella's Kitchen Pouches", 'Empanadas', 'Extra Salami or Feta',
		'Focaccia', 'Frittata', 'Fudge', 'Granola', 'Honey', 'Jam','Kids biscuit',
		'Lemon and coconut', 'Muesli', 'Muffin', 'Raspberry shortbread sandwich',
		'Pastry', 'Salad', 'Sandwich','Scandinavian', 'Scone','Tacos/Fajita', 
		'Toast', 'Vegan Feast','Vegan mincepie', 'Victorian Sponge']
		
    drink = ['Coffee','Coke', 'Coffee granules ', 'Hot chocolate',  'Juice', 'Mighty Protein',
		 'Mineral water',  'My-5 Fruit Shoot',  'Smoothies', 'Soup', 'Tea']
    
    # Assigning Category to items based on items in food and drink list    
    df['Category'] = df['Item'].apply(lambda x :
                                      'Food' if x in food else 
                                      ('Drink' if x in drink else 'Unknown'))
    
    # Calculating Average food and drink price    
    item_avg_price = df.groupby('Category')['Item_Price'].mean() 
    print('Average price of Drink: ' + '$' + str( round(item_avg_price['Drink'], 2) ))
    print('Average price of Food : ' + '$' + str( round(item_avg_price['Food'], 2) ))  
    
    
    #------------------ Ques-6 ------------------------------
    print('\n********** Question 6 ********** ')  
    
    item_revenue = df.groupby('Category')['Item_Price'].sum()
    print('Coffee shop makes more revenue by selling {}'
          .format('Drinks.' if item_revenue['Drink'] > item_revenue['Food'] else 'Food.'))
    
    
    #------------------ Ques-7 & 8 ------------------------------
    print('\n********** Question 7 & 8 ********** ')  
    
    # Get the unique counts of items group by weekday
    item_counts = df.groupby('Weekday')['Item'].value_counts()
    
    # Get the top five counts items group by weekday from item_counts
    top_five_items = item_counts.groupby('Weekday').nlargest(5)
    
    # Get bottom five popular items 
    bottom_five_items = item_counts.groupby('Weekday').nsmallest(5)
    
    weekday_list = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', \
                    'Thursday', 'Friday', 'Saturday']
    
    # Ques -7 Conclusion    
      
    print('\nMost favorable items per week: \n')    
    for day in weekday_list :
        print(top_five_items[day].to_string(),'\n')
                
    print('\nConclusion: \nThe most favourable items are: Coffee, Bread, Tea, Cake. '
          '\nThese 3 items are common for each day of the week.'
          ' Other than these, Pastry and  Sandwich were also bought for 2-3 days in a week.\n')    
    
    print('\nLeast favorable items per week: \n')
    # Ques -8 Conclusion
    for day in weekday_list :
        print(bottom_five_items[day].to_string(), '\n')
    
    print('\nConclusion: \nThe list is not the same for least favourable items each day of week.'
          '\nFrom the above list, only Chocolate is the item which is common for almost 6 days.'
         '  While for the other items, the list changes.')
    
    #------------------ Ques-9 ------------------------------
    
    # Total transactions and total drinks sold
    total_transactions = df['Transaction'].count()
    
    total_drinks_sold = df.loc[df['Category'] == 'Drink']['Transaction'].count()
    
    # Calculating drinks per transaction
    drinks_per_trans = total_drinks_sold / total_transactions
    
    print('\n********** Question 9 ********** ') 
    print('Number of drinks per transaction: ', round(drinks_per_trans, 1))
    
    
    
except Exception as e:
    print(e)
    print('Unable to read input file.')    