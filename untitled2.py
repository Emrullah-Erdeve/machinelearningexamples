# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 21:23:03 2022

@author: Emrullah erdeve
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris=load_iris()

x=iris.data
y=iris.target

x=(x-np.min(x)/(np.max(x)-np.min(x)))

#%%
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
 
 #%%
from sklearn.neighbors import KNeighborsClassifier
 
knn=KNeighborsClassifier(n_neighbors=10)
#%%
from sklearn.model_selection import cross_val_score

accuarices=cross_val_score(estimator=knn,X=x_train,y=y_train,cv=10)
print("average accuracy:",np.mean(accuarices))
print("average accuracy:",np.std(accuarices))

#%%
from sklearn.model_selection import GridSearchCV
grid={"n_neighbors":np.arange(1,50)}
knn= KNeighborsClassifier()
knn_cv=GridSearchCV(knn,grid,cv=10)
knn_cv.fit(x,y)
#%%
print("tuned hyperparamater:",knn_cv.best_params_)
print("tuned hyperparamater:",knn_cv.best_score_)
# %% Grid search CV with logistic regression

x = x[:100,:]
y = y[:100] 

from sklearn.linear_model import LogisticRegression

grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}  # l1 = lasso ve l2 = ridge

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv = 10)
logreg_cv.fit(x,y)

print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)
print("accuracy: ",logreg_cv.best_score_)







