# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 22:53:05 2022

@author: fthsl
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple_linear_regression_dataset.csv", sep=';')

x = df.iloc[:, [0,2]].values

y = df.maas.values.reshape(-1,1)

multiple_linear_reg = LinearRegression()
multiple_linear_reg.fit(x, y)


print("b0: ",multiple_linear_reg.intercept_)
print("b1: ",multiple_linear_reg.coef_)


print(multiple_linear_reg.predict(np.array([[10,35], [5,35]])))

