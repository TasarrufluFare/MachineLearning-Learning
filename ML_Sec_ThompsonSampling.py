import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

#UpperConfidenceBound ile Reklam Göstermek 
import random
#Üretilecek sayı adeti
N = 10000
# Her ilan için 10 secenekten birini secerken kullanıyorum
d = 10
toplam = 0 # Toplam Ödül
secilenler = []
birler = [0] * d
sifirlar = [0] * d
for ilan_numarasi in range(0, N):
    ilan = 0
    max_th = 0
    for i in range(0, d): # Her ilan olasılığını kontrol ediyorum
        rasgele_beta = random.betavariate(birler[i] + 1, sifirlar[i] + 1)
        if rasgele_beta > max_th:
            max_th = rasgele_beta
            ilan = i
        
    
    secilenler.append(ilan)
    odul = veriler.values[ilan_numarasi, ilan] # Verilerdeki secilen 1 ise ödül 1, secilen 0 ise ödül 0
    if odul == 1:
        birler[ilan] = birler[ilan] + 1
    else:
        sifirlar[ilan] = sifirlar[ilan] + 1
    toplam = toplam + odul
    
plt.hist(secilenler)
plt.show() 
print(toplam)
    
    
    
    
    
    
    
    
    
    