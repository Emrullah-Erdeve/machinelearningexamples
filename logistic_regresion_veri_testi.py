
"""
Created on Thu Feb 17 15:32:06 2022

@author: Emrullah erdeve
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data.drop(["Unnamed: 32","id"],axis=1,inplace = True)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
print(data.info())

y = data.diagnosis.values
x_ = data.drop(["diagnosis"],axis=1)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_,y,test_size = 0.5,random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("test accuracy {}".format(lr.score(x_test,y_test)))


from sklearn.metrics import confusion_matrix

tahmin=lr.predict(x_test)
confusion_matrix(y_test,tahmin)

yeni_veri=np.array([[16.29, 14.34,175.1, 1500, 0.2003,0.23,0.198,0.1043,0.1809,0.05883,1.5,1.7813,5.438,94.44,0.01149,0.02461,0.1688,0.01885,0.01756,0.005115,22.54,16.67,152.201,1575.00001,0.1374,0.505,0.6,0.25,0.2364,0.07678]])
    

lr.predict(yeni_veri)