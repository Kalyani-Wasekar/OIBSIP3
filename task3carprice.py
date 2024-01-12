##########  CAR PRICE PREDICTION  ##########

print('CAR PRICE PREDICTION')

#Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Load the dataset
dataset = pd.read_csv('task3carpricedata.csv')

#Display the first few rows of the dataset
print(dataset)

#Select relevant features
X = dataset[['Year', 'Present_Price', 'Driven_kms', 'Owner']]
y = dataset['Selling_Price']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test)

#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

#Make a prediction for a new set of features
new_data = np.array([[2017, 10.0, 50000, 1]])
prediction = model.predict(new_data.reshape(1, -1))
print(f'Predicted Selling Price: {prediction[0]}')

#Visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Actual vs Predicted Selling Price')
plt.show()
