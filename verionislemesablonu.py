
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')

veriler.cinsiyet = [1 if each == "e" else 0 for each in veriler.cinsiyet]

#pd.read_csv("veriler.csv")
#test
from sklearn.model_selection import train_test_split
x=veriler.iloc[:,1:4].values
y=veriler.iloc[:,4:].values
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr=LogisticRegression()
logr.fit(X_train,y_train)
y_pred=logr.predict(X_test)

list=[]

for each in y_pred:
    if(each==y_test):
        list.append("doğru bildi")

    else:
        list.append("0")



