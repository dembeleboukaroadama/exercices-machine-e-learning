# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:19:22 2020

@author: Boukaro Adama
"""
import pandas as pd

effectifs = data["quart_mois"].value_counts()
modalites = effectifs.index # l'index de effectifs contient les modalités
tab = pd.DataFrame(modalites, columns = ["quart_mois"]) # création du tableau à partir des modalités
tab["n"] = effectifs.values
tab["f"] = tab["n"] / len(data) # len(data) renvoie la taille de l'échantillon
tab = tab.sort_values("quart_mois") # tri des valeurs de la variable X (croissant)
tab["F"] = tab["f"].cumsum() # cumsum calcule la somme cumulée


for cat in data["categ"].unique():
subset = data[data.categ == cat] # Création du sous-échantillon
print("-"*20)
print(cat)
print("moy:\n",subset['montant'].mean())
print("med:\n",subset['montant'].median())
print("mod:\n",subset['montant'].mode())
print("var:\n",subset['montant'].var(ddof=0)) #est utilise pour avoir les memes valeurs que lors du calcul à la main
print("ect:\n",subset['montant'].std(ddof=0))
print("skw:\n",subset['montant'].skew())    #Le Skewness empirique
print("kur:\n",subset['montant'].kurtosis()  #Le Kurtosis empirique


subset["montant"].hist() # Crée l'histogramme
plt.show() # Affiche l'histogramme


import numpy as np
depenses = data[data['montant'] < 0]
dep = -depenses['montant'].values
n = len(dep)
lorenz = np.cumsum(np.sort(dep)) / dep.sum()
lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0
plt.axes().axis('equal')
xaxis = np.linspace(0-1/n,1+1/n,n+1) #Il y a un segment de taille n pour chaque individu, plus 1
segment supplémentaire d'ordonnée 0. Le premier segment commence à 0-1/n, et le dernier termine à
1+1/n.
plt.plot(xaxis,lorenz,drawstyle='steps-post')
plt.show()
        
AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n # Surface sous la courbe de Lorenz. Le premier
segment (lorenz[0]) est à moitié en dessous de 0, on le coupe donc en 2, on fait de même pour le
dernier segment lorenz[-1] qui est à moitié au dessus de 1.
S = 0.5 - AUC # surface entre la première bissectrice et le courbe de Lorenz
gini = 2*S
gini        
        

#representez la covariance
      
import matplotlib.pyplot as plt
depenses = data[data.montant < 0]
plt.plot(depenses["solde_avt_ope"],-depenses["montant"],'o',alpha=0.5)
plt.xlabel("solde avant opération")
plt.ylabel("montant de dépense")
plt.show()

python
import scipy.stats as st
import numpy as np
print(st.pearsonr(depenses["solde_avt_ope"],-depenses["montant"])[0])
print(np.cov(depenses["solde_avt_ope"],-depenses["montant"],ddof=0)[1,0]      
        
        
taille_classe = 500 # taille des classes pour la discrétisation
groupes = [] # va recevoir les données agrégées à afficher
# on calcule des tranches allant de 0 au solde maximum par paliers de taille taille_classe
tranches = np.arange(0, max(depenses["solde_avt_ope"]), taille_classe)
tranches += taille_classe/2 # on décale les tranches d'une demi taille de classe
indices = np.digitize(depenses["solde_avt_ope"], tranches) # associe chaque solde à son numéro de
classe
for ind, tr in enumerate(tranches): # pour chaque tranche, ind reçoit le numéro de tranche et tr la
tranche en question
montants = -depenses.loc[indices==ind,"montant"] # sélection des individus de la tranche ind
if len(montants) > 0:
g = {
'valeurs': montants,
'centre_classe': tr-(taille_classe/2),
'taille': len(montants),
'quartiles': [np.percentile(montants,p) for p in [25,50,75]]
}
groupes.append(g)
# affichage des boxplots
plt.boxplot([g["valeurs"] for g in groupes],
positions= [g["centre_classe"] for g in groupes], # abscisses des boxplots
showfliers= False, # on ne prend pas en compte les outliers
widths= taille_classe*0.7, # largeur graphique des boxplots
)
# affichage des effectifs de chaque classe
for g in groupes:
plt.text(g["centre_classe"],0,"(n=
{})".format(g["taille"]),horizontalalignment='center',verticalalignment='top')
plt.show()
# affichage des quartiles
for n_quartile in range(3):
plt.plot([g["centre_classe"] for g in groupes],
[g["quartiles"][n_quartile] for g in groupes])
plt.show()  


import datetime as dt
# Selection du sous-échantillon
courses = data[data.categ == "COURSES"]
# On trie les opérations par date
courses = courses.sort_values("date_operation")
# On ramène les montants en positif
courses["montant"] = -courses["montant"]
# calcul de la variable attente
r = []
last_date = dt.datetime.now()
for i,row in courses.iterrows():
days = (row["date_operation"]-last_date).days
if days == 0:
r.append(r[-1])
else:
r.append(days)
last_date = row["date_operation"]
courses["attente"] = r
courses = courses.iloc[1:,]
# on regroupe les opérations qui ont été effectués à la même date
# (courses réalisées le même jour mais dans 2 magasins différents)
a = courses.groupby("date_operation")["montant"].sum()
b = courses.groupby("date_operation")["attente"].first()
courses = pd.DataFrame({"montant":a, "attente":b})




import statsmodels.api as sm
Y = courses['montant']
X = courses[['attente']]
X = X.copy() # On modifiera X, on en crée donc une copie
X['intercept'] = 1.
result = sm.OLS(Y, X).fit() # OLS = Ordinary Least Square (Moindres Carrés Ordinaire)
a,b = result.params['attente'],result.params['intercept']


plt.plot(courses.attente,courses.montant, "o")
plt.plot(np.arange(15),[a*x+b for x in np.arange(15)])
plt.xlabel("attente")
plt.ylabel("montant")
plt.show()
        

"""le coefficient de correlation"""

import scipy.stats as st
import numpy as np

st.pearsonr(var1,var2)
np.cov(car1,var2)

"""le chi-2"""

X=[]
Y=[]

"""le tableu de contaigeance"""

c=data[[X,Y]].pivo_table(index=X, columns=Y, aggfuc=len)
count=c.copy()

tx=data[X].value_counts()
ty=data[Y].value_counts()

count.loc[:,"total"]=tx
count.loc["total",:]=ty

count.loc["lotal","total"]=len(data)
count

indep=tx.dot(tx.ty)/n

mesur=(count-indep)**2/indep

sum_m=mesure.sum().sum()
sns.heatmap(mesure/sum_m, annot=c)



























        
        
        
        
        
        
        
