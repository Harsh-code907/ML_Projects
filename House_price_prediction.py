import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("data.csv")

X = dataset[["SqFt","Bedrooms","Bathrooms"]]
y = dataset["Price"]


X_train,X_test,y_train,y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

mse = mean_squared_error(y_test,y_predict)


SqFt =int(input("Enter SquareFt:")) 
Bedrooms = int(input("Enter Bedrooms:")) 
Bathrooms = int(input("Bathrooms:")) 

# Testing values for prediction.
new_house = [[SqFt, Bedrooms, Bathrooms]]
predicted_price = model.predict(new_house)
print("Predicted Price:", predicted_price[0])

#Calculating Price by mathematical formula
coef = model.coef_
intercept = model.intercept_
price = (coef[0] * SqFt) + (coef[1] * Bedrooms) + (coef[2] * Bathrooms) + intercept
print("Predicted Price (manual):", round(price, 2))