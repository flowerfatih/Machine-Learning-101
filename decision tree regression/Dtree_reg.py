# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 23:38:12 2022

@author: fthsl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("decision+tree+regression+dataset.csv", sep=';', header= None)

x = df.iloc[:, 0].values.reshape(-1,1)
y = df.iloc[:, 1].values.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(x, y)
x_ = np.arange(min(x), max(x), 0.01).reshape(-1,1)
y_head = tree_reg.predict(x_)

print(tree_reg.predict([[6]]))


#%% visualize

plt.scatter(x, y, color='red')
plt.plot(x_, y_head, color='green')
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()


