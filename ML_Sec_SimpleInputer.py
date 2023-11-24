# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("eksikveriler.csv")

print(veriler)

boy = veriler [['boy']]
print(boy)

boykilo = veriler[["boy","kilo"]]
print(boykilo)



#eksik veriler yerine diger verilerin ortalamasını yazmak
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

yas = veriler.iloc[:,1:4].values
print("Yas Degiskeni")
print(yas)


yas = my_imputer.fit(yas[:,:]).transform(yas[:,:])
print("Yeni Veriler")
print(yas)


#Kategorileştirme Binominal (Kategorik ---> Numeric)
from sklearn import preprocessing
ulke = veriler.iloc[:,0:1].values
print("Ulke Degiskeni")
print(ulke)

le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print("Ulke Degiskeni Numaralandirilmis")
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print("Ulke Degiskeni Numaralandirilmis Binominal")
print(ulke)


#Verileri DataFramelerde birleştirmek
sonuc = pd.DataFrame(data=ulke, index = range(22), columns=["fransa", "türkiye", "amerika"])
print(sonuc)

sonuc2 = pd.DataFrame(data= yas, index = range(22), columns=["boy","kilo","yaş"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
sonuc3 = pd.DataFrame(data=cinsiyet, index = range(22), columns=["cinsiyet"])
print(sonuc3)

s = pd.concat([sonuc,sonuc2],axis=1)

print(s)

s2 = pd.concat([s,sonuc3],axis=1)

print(s2)



#Verileri Test ve Train Olarak Bölmek
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test  = train_test_split(s, sonuc3, test_size = 0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.fit_transform(x_test)



