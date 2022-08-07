# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 16:40:36 2022

@author: fthsl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

plt.scatter(M.radius_mean, M.texture_mean, color="red", label="kotu")

plt.scatter(B.radius_mean, B.texture_mean, color="green", label="iyi")
plt.xlabel("radius_mean")
plt.ylabel("texture mean")
plt.legend()
plt.show()


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(['diagnosis'], axis=1)

#normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)

print("KNN score: {}".format(knn.score(x_test, y_test)))

score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train, y_train)
    score_list.append(knn2.score(x_test, y_test))
    
    
plt.plot(range(1,15), score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
#en basarili sonuc k degerini 4 verdigimizde elde ediliyor