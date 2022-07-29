# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt


#%%
df = pd.read_csv("linear_regression_dataset.csv", sep=";")
plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()


'''
residual = y - y_head
MSE(mean squared error) = sum(residual^2)/n (n=number of samples)


'''

#%%
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)


#%%
b0 = linear_reg.predict([[0]])

print(b0)

b0 = linear_reg.intercept_

print(b0)

b1 = linear_reg.coef_
print(b1)


