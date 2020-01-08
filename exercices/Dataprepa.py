# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:05:07 2019

@author: Boukaro Adama
"""

import statsmodels as stats
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dtst=pd.read_csv('C:\DATA ANALYSE\REP_MLEARNING\credit_immo.csv')
X=dtst.iloc[:,-9:-1].values
Y=dtst.iloc[:,-1].values

"""la variable y est une variable expliquée (independants)
alors que la variable x est une variable explicative """


"""netoyage des donnees"""
from sklearn.preprocessing import Imputer
imp=Imputer() 
"""la strategy  est ce qu on doit retourner dans a la place des valeurs nulls(dans notre cas la moyenne)
apres quoi on applele la fonction fit de imputer et enfin la fonction transform"""
imp.fit(X[:,0:1])
imp.fit(X[:,7:8])
X[:,0:1]=imp.transform(X[:,0:1])
X[:,7:8]=imp.transform(X[:,7:8])


"""encodage des donnees"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lab_X=LabelEncoder()    
X[:,2]=lab_X.fit_transform(X[:,2])
X[:,5]=lab_X.fit_transform(X[:,5])
oneHot=OneHotEncoder(categorical_features=[2])
oneHot=OneHotEncoder(categorical_features=[5])
X=oneHot.fit_transform(X).toarray()

lab_y=LabelEncoder()
Y=lab_y.fit_transform(Y)

"""enchantillonage"""

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_tran,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=0)


"""mise de donnees au meme echelle"""
from sklearn.preprocessing import normalize
X_tran = normalize(X_tran)
X_test = normalize(X_test)


"""model de regression lineaire"""
dtst2=pd.read_csv('lycee.csv')

x=dtst2.iloc[:,:-1].values
y=dtst2.iloc[:,-1].values

#echantillonnage
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#creer un model de regression lineaire

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

#faire une prediction

progression=reg.predict(x_test)


plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, reg.predict(x_train), color='green')
plt.xlabel('nombre d heure de revison')
plt.ylabel('note en %')
plt.title('prevision des eleves')
plt.show()


#regression multiple

dtst3=pd.read_csv('EU_I_PIB.csv')
x=dtst3.iloc[:,-4:].values
y=dtst3.iloc[:,-5].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lb=LabelEncoder()
x[:,0]=lb.fit_transform(x[:,0])
on=OneHotEncoder(categorical_features=[0])
x=on.fit_transform(x).toarray()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

#☻construction du model de regression

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train, y_train)

y_pre=reg.predict(x_test)

reg.predict(np.array([[0,1,145790.4,120266.85,384897.62]]))







