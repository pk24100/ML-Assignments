import pandas as panda
import numpy as npy
import sympy as sym
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from statistics import mean, variance
import matplotlib.pyplot as plot

purchase_data = panda.read_excel(r'C:\Users\Win10\Downloads\ML Assign\Lab Session1 Data.xlsx',usecols="A:E")
print(purchase_data)

A = purchase_data.iloc[:, 1:4]  
C = purchase_data.iloc[:, 4:5]  

A= A.to_numpy()
C= C.to_numpy()

#A1
print("Matrix A:")
print(A)
print("Matrix C:")
print(C)


print(f'Dimensionality of vector space: {A.shape[1]}')
print(f'No of vectors: {A.shape[0]}')
print(f'Rank - {sym.Matrix(A).rank()}')

X = npy.linalg.pinv(A) @ C
print(f'Cost of each product: {X}')

#A2
new_customer = [[5,15,20]]
total_purchase = npy.dot(new_customer,X)
print(f'Amount paid by customer if [5,15,20]: {total_purchase}')

#A3
purchase_data['Category'] = purchase_data['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')

Y= purchase_data.iloc[:,1:4]
Z= purchase_data['Category']

Y_train, Y_test, Z_train, Z_test = train_test_split(Y,Z,test_size=0.25,random_state=0)
classifier = DecisionTreeClassifier()
classifier.fit(Y_train,Z_train)

print(f'Accuracy: {classifier.score(Y_test,Z_test)}')

new_customer = [[5,10,55]]
result= classifier.predict(new_customer)
print(f'New customer is: {result[0]}')


#A4- 
irctc_stock_details = panda.read_excel(r'C:\Users\Win10\Downloads\ML Assign\Lab Session1 Data.xlsx',sheet_name='IRCTC_Stock_Price')
print(irctc_stock_details)

print(f"Mean of price - {mean(irctc_stock_details['Price'])}")
print(f"Variance of price - {variance(irctc_stock_details['Price'])}")

wednesdays = irctc_stock_details[irctc_stock_details['Day']=='Wed']
print(f"Mean of price of all wednesdays - {mean(wednesdays['Price'])}")
print(f"Mean of price of all wednesdays is {mean(irctc_stock_details['Price'])- mean(wednesdays['Price'])} less than mean of price of all days")

april = irctc_stock_details[irctc_stock_details['Month']=='Apr']
print(f"Mean of price in april - {mean(april['Price'])}")
print(f"Mean of all prices is { mean(april['Price']) - mean(irctc_stock_details['Price'])} less than mean of all price ")

negative_change = len(irctc_stock_details[irctc_stock_details['Chg%']<0])
all_rows=  len(irctc_stock_details)
print(f"Probability of loss - {negative_change/all_rows}")

wednesday_with_profit = len(wednesdays[wednesdays['Chg%']>0])
print(f"Probability of profit on Wed - {wednesday_with_profit/all_rows}")

print(f"Conditional probability of making profit, given that today is Wednesday - {((wednesday_with_profit/all_rows)/(len(wednesdays)/all_rows))}")

days = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}

day_number = irctc_stock_details['Day'].apply(lambda x: days[x])

plot.scatter(day_number,irctc_stock_details['Chg%'])
plot.xticks(list(days.values()), list(days.keys()))

plot.xlabel('Day of the Week')
plot.ylabel('Chg%')
plot.title('Scatter Plot of Chg% Against the Day of the Week')
plot.show()