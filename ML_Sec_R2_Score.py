# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#R2 Import Ediliyor
from sklearn.metrics import r2_score


veriler = pd.read_csv("maaslar.csv")

#Girdi ve Çıktıların Oluşturulması (Data Sliceing)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:3]

#Numpy Array Oluşturulması
x_values = x.values
y_values = y.values

print(veriler)
print(x_values)
print(y_values)


# LinearRegression İle Tahmin Etmek
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_values, y_values)


#Polynomial Regression Uygulanması
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x_values)
print(x_poly)
lin_reg_new = LinearRegression()
lin_reg_new.fit(x_poly,y_values)


# Support Vector Machine
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_values_scaled = sc1.fit_transform(x_values)
sc2 = StandardScaler()
y_values_scaled = sc2.fit_transform(y_values)
y_values_scaled = y_values_scaled.flatten()
from sklearn.svm import SVR
svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_values_scaled, y_values_scaled)


#Tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[3.3]]))

print(lin_reg_new.predict(poly_reg.fit_transform([[11]])))
print(lin_reg_new.predict(poly_reg.fit_transform([[3.3]])))

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[3.3]]))



#Görselleştirme
plt.scatter(x_values,y_values, color="red")
plt.plot(x,lin_reg.predict(x_values), color = "blue")
plt.show()

plt.scatter(x_values, y_values, color="red")
plt.plot(x_values,lin_reg_new.predict(x_poly),color="green")
plt.show()

plt.scatter(x_values_scaled,y_values_scaled, color="red")
plt.plot(x_values_scaled,svr_reg.predict(x_values_scaled), color = "orange")
plt.show()



#Decision Tree İle Tahmin Algoritması
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x_values, y_values)

plt.scatter(x_values, y_values, color="pink")
plt.plot(x_values, dt_reg.predict(x_values))
plt.show()


#Random Forest ile tahmin etme
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(x_values,y_values.ravel())
print(rf_reg.predict([[6.6]]))

plt.scatter(x_values,y_values,color="red")
plt.plot(x_values,rf_reg.predict(x_values),color="blue")
plt.plot(x_values,rf_reg.predict(x_values + 0.5), color="green")
plt.plot(x_values,rf_reg.predict(x_values - 0.7), color="yellow")

print("Lineer Regression R2 Değeri")
print(r2_score(y_values, lin_reg.predict(x_values)))

print("Polynomial Regression R2 Değeri")
print(r2_score(y_values, lin_reg_new.predict(poly_reg.fit_transform(x_values))))

print("SVR R2 Değeri")
print(r2_score(y_values_scaled, svr_reg.predict(x_values_scaled)))

print("Decision Tree R2 Değeri")
print(r2_score(y_values, dt_reg.predict(x_values)))
print(r2_score(y_values, dt_reg.predict(x_values +0.4)))
print(r2_score(y_values, dt_reg.predict(x_values-0.4)))

print("Random Forest R2 Değeri")
print(r2_score(y_values, rf_reg.predict(x_values)))
print(r2_score(y_values, rf_reg.predict(x_values +0.4)))
print(r2_score(y_values, rf_reg.predict(x_values-0.4)))
