# Apriori Algoritması

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori

veriler = pd.read_csv("sepet.csv", header=None)

t = []
for i in range(0,7501):
    t.append([str(veriler.values[i,j]) for j in range(0,20)])
    
    
rules = apriori(t, min_support = 0.01, min_confidence = 0.2, min_lift = 3, 
        min_lenght = 2)

print(list(rules))
    
