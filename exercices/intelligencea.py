# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:32:26 2020

@author: Boukaro Adama
"""

import numpy as np
import pandas as pd
import matplotlib as ptl
from matplotlib import pyplot

from collections import Counter

dataset= pd.read_csv('NLP_MLP_last.csv')

titres=["Discover what your friends really think of you, The GNU Make Book Early Access,The GNU Make Book Early Access",
        "The GNU Make Book Early Access,The GNU Make Book Early Access,The GNU Make Book Early Access",
        "The GNU Make Book Early Access,The GNU Make Book Early Access,The GNU Make Book Early Access",
        "The GNU Make Book Early Access,The GNU Make Book Early Access,The GNU Make Book Early Access",
        "The GNU Make Book Early Access,The GNU Make Book Early Access,The GNU Make Book Early Access"]

#fonction pour trouver les mots uniques dans les titres

mots_unique=list(set(" ".join(titres).split(" ")))

def make_matrix(titres, vocab):
    matrix=[]
    #compter chaque mot du titre et en faire une matrice
    
    for titre in titres:
        counter= Counter(titre)
        
        #transformer le dictionnaire en ligne matricielle en utulisnt le vocab
        
        row=[counter.get(w,0) for w in vocab]
        matrix.append(row)    
        
        df=pd.DataFrame(matrix)
        df.columns=mots_unique
        return df
    
print(make_matrix(titres, mots_unique))


import re

nv_titre=[re.sub(r'[^\w\s\d]', '', h.lower()) for h in titres]
nv_titre=[re.sub(r'[\s+]', ' ', h) for h in titres]
mots_unique=list(set("".join(nv_titre).split(" ")))
    
print(make_matrix(titres, mots_unique))


#creation de la matrice avec vectoriseur

from sklearn.feature_extraction.text import CountVectorizer

vectoriseur= CountVectorizer(lowercase=True,stop_words='english')
matrix=vectoriseur.fit_transform(titres)

print(matrix.todense())

dataset['full']=dataset['titre']+" "+dataset['url']
full_matrix=vectoriseur.fit_transform(dataset['full'].values.astype("U"))

print(full_matrix.shape)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    