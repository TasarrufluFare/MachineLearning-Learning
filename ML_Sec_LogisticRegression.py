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


veriler = pd.read_csv("veriler.csv")

print(veriler)

#Ülke Kolonunun çıkartılması
veriler = veriler.drop("ulke",axis=1)
#veriler = veriler[veriler["yas"]>=20]
print(veriler)

#Değişkenlerin Tanımları
x = veriler.iloc[:,:3]
y = veriler.iloc[:,3:]
y = y.to_numpy().ravel()

#Tahmin Modelinin Oluşturulması
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state=0)

#Verilerin Ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=0)
log_reg.fit(x_train_scaled, y_train)

y_pred = log_reg.predict(x_test_scaled)


#Confusion Matrix Oluşturmak
from sklearn.metrics import confusion_matrix

print("Logistic Regression Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

#En Yakın Komşu Algoritması
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")
knn.fit(x_train_scaled,y_train)

y_pred = knn.predict(x_test_scaled)

print("KNN Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Support Vector Classifier Algoritması
from sklearn.svm import SVC

svc_reg = SVC(kernel="rbf")
svc_reg.fit(x_train_scaled,y_train)

y_pred = svc_reg.predict(x_test_scaled)

print("SVC Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)


