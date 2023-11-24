# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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



plt.scatter(x_values, y_values, color="red")
plt.plot(x_values,lin_reg_new.predict(x_poly),color="green")
plt.show()

plt.scatter(x_values_scaled,y_values_scaled, color="red")
plt.plot(x_values_scaled,svr_reg.predict(x_values_scaled), color = "orange")



