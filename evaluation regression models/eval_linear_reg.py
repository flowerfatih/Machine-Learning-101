# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 00:09:21 2022

@author: fthsl
"""

import pandas as pd



#%%
df = pd.read_csv("linear_regression_dataset.csv", sep=";")
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

y_head = linear_reg.predict(x)
#%%
from sklearn.metrics import r2_score

print("r_score: ", r2_score(y, y_head))