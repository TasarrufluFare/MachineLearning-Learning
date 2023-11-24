import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

'''
#Radndom Selection ile Reklam Göstermek - 1231 ödül kazanıldı
#Random sayı üretip seçim yapacağız
import random

#Üretilecek sayı adeti
N = 10000
# Her ilan için 10 secenekten birini secerken kullanıyorum
d = 10

toplam = 0
secilenler = []
for ilan_numarasi in range(0, N):
    ilan = random.randrange(d)
    secilenler.append(ilan)
    odul = veriler.values[ilan_numarasi, ilan] # Verilerdeki secilen 1 ise ödül 1, secilen 0 ise ödül 0
    toplam = toplam + odul
    
plt.hist(secilenler)
plt.show()
'''

#UpperConfidenceBound ile Reklam Göstermek 
import random
import math
#Üretilecek sayı adeti
N = 10000
# Her ilan için 10 secenekten birini secerken kullanıyorum
d = 10
odüller = [0] * d #10 Elemanlı ve bütün elemanlar 0
toplam = 0 # Toplam Ödül
tiklama_sayisi = [0] * d # O ana kadarki tıklama sayısı, her ilan için
secilenler = []
for ilan_numarasi in range(0, N):
    ilan = 0
    max_ucb = 0
    for i in range(0, d): # Her ilan olasılığını kontrol ediyorum
        #Sıfıra bölünme durumu kontrolü
        if tiklama_sayisi[i]>0:
            ortalama = odüller[i]/tiklama_sayisi[i]
            delta = math.sqrt((3/2)* math.log(ilan_numarasi)/tiklama_sayisi[i]) 
            ucb = ortalama + delta
        else:
            ucb = N*10
        if max_ucb < ucb: # Yeni maximumu belirlemek
            max_ucb = ucb
            ilan = i
    
    secilenler.append(ilan)
    tiklama_sayisi[ilan] = tiklama_sayisi[ilan] + 1
    
    odul = veriler.values[ilan_numarasi, ilan] # Verilerdeki secilen 1 ise ödül 1, secilen 0 ise ödül 0
    odüller[ilan] = odüller[ilan] + odul
    toplam = toplam + odul
    
plt.hist(secilenler)
plt.show() 
print(toplam)
    
    
    
    
    
    
    
    
    
    