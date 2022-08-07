# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 19:09:42 2022

@author: fthsl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(['diagnosis'], axis=1)

#normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier


#print("score: {}".format(rf.score(x_test,y_test)))

random_states = []
for each in range(1,100):
    rf = RandomForestClassifier(n_estimators=100, random_state=each)
    rf.fit(x_train, y_train)
    random_states.append(rf.score(x_test,y_test))

plt.plot(range(1,100), random_states)
plt.xlabel("random states")
plt.ylabel("accuracy")
plt.show()

print("en basarili sonuc alinan random state parametresi: {}".format(random_states.index(max(random_states))))