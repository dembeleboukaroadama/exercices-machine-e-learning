# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:30:36 2020

@author: Boukaro Adama
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as ptl
import seaborn as sbrn

data=pd.read_csv('Social.csv')

x=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
std = StandardScaler()

x_train=std.fit_transform(x_train)
x_test=std.fit_transform(x_test)

#construction de l'arbre de decidion

from sklearn.tree import DecisionTreeClassifier

classification=DecisionTreeClassifier(criterion='entropy', random_state=0)
classification.fit(x_train, y_train)

#prediction

y_pre=classification.predict(x_test)
"""ici nous avons cree un model de classification"""

#contruction de la matrix de confusion

from sklearn.metrics import confusion_matrix

matrixe = confusion_matrix(y_pre, y_test)

#visualisation







































