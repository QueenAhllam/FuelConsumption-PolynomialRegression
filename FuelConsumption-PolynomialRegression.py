import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("FuelConsumption.csv")
x_1 = data.iloc[:,4:5].values
x_2 = data.iloc[:,11:12].values
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(x_2, y, shuffle=True, test_size=0.15, random_state=42)

poly = PolynomialFeatures(degree=5)
x_train_new = poly.fit_transform(X_train)
x_test_new = poly.fit_transform(X_test)

model = LinearRegression()
model.fit(x_train_new, y_train)
y_hat = model.predict(x_test_new)

xx = np.linspace(10, 60)
yy = model.intercept_ + model.coef_[1] * np.power(xx, 1) + model.coef_[2] * np.power(xx, 2) + model.coef_[3] * np.power(xx, 3) + model.coef_[4] * np.power(xx, 4) +model.coef_[5] * np.power(xx, 5) 

plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.plot(xx , yy)
plt.show()


👏 This code belongs to **Queen Ahlam**! 👑  
A small masterpiece that reflects her creativity and skills! 🚀✨

