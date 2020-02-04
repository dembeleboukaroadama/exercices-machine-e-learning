# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 22:23:56 2020

@author: Boukaro Adama
"""

from sklearn.datasets import load_iris
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

iris=load_iris()

x=iris.data
y=iris.target

names=list(iris.target_names)

print(f'x contient {x.shape[0]} exemples et {x.shape[1]} variables')
print(f'il ya {np.unique(y).size} classes')

plt.scatter(x[:,0], x[:,1], c=y, alpha=0.5, s=x[:, 2]*100)
""" c specifie la couleurs, alpha la transparence, s la grandeur des points"""

"""graphique 3d"""
from mpl_toolkits.mplot3d import Axes3D
ax=plt.axes(projection='3d')
ax.scatter(x[:,0], x[:,1],x[:,2], c=y)

f= lambda x,y: np.sin(x) + np.cos(x+y)

X=np.linspace(0,5,100)
Y=np.linspace(0,5,100)
X,Y=np.meshgrid(X,Y)
Z=f(X, Y)
Z.shape
ax=plt.axes(projection='3d')
ax.plot_surface(X,Y,Z, cmap="plasma")


"""les histogrammes"""
plt.hist(x[:,0], bins=15)

"""bins precise le nombre de sections deans notre histogramme
on peut mettre deux variables dans un histogramme"""

plt.hist2d(x[:, 0],x[:,1])
plt.xlabel('longuer')
plt.ylabel("largeur")
plt.colorbar()

"""contour plot"""

plt.contour(X,Y,Z, colors="black")
plt.contourf(X,Y,Z, cmap="RdGy")
plt.colorbar() """permet d'ajouter une lengende"""


"""l'imshow()"""

plt.imshow(np.corrcoef(x.T))
plt.colorbar()
plt.imshow(Z)
plt.colorbar()

iris.to_csv("iris.csv")



