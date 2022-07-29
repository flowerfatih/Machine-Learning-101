# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 23:07:44 2022

@author: fthsl
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial+regression.csv", sep=';')


y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x, y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
plt.show()

#linear reg => y = b0 + b1*x
#multiple reg => y = b0 + b1*x1 + b2*x2

#%%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

#%%
#predict

y_head = lr.predict(x)

plt.plot(x, y_head, color='red', label='linear')

print("1 milyon liralik arabanin hizi: ",lr.predict([[10000]]))
#%%

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree=2)

x_polynomial = polynomial_regression.fit_transform(x)

linear_reg2 = LinearRegression()
linear_reg2.fit(x_polynomial, y)

#%%

y_head2 = linear_reg2.predict(x_polynomial)
plt.plot(x,y_head2,color='green', label='poly')
plt.legend()


#%%
polynomial_regression = PolynomialFeatures(degree=4)

x_polynomial = polynomial_regression.fit_transform(x)

linear_reg2 = LinearRegression()
linear_reg2.fit(x_polynomial, y)

#%%

y_head2 = linear_reg2.predict(x_polynomial)
plt.plot(x,y_head2,color='black', label='poly')
plt.legend()
plt.show()

















