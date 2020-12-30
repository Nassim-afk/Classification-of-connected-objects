
"""
Created on Fri Dec 27 12:22:15 2019

"""

#-------------Projet2:Classification ds objets connectes-------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
#-----------------Partie 1 -------

data=pd.read_csv('datasetFlows.csv') #Lecture des données

Y = data["type"] #Les étiquettes
#print (Y)   
X = data.drop(labels="label", axis=1).drop(labels="type", axis=1)
#print(X)

#Séparation des données en un set de test et d'apprentissage 
X_train,X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.3,random_state=10)

#*****************Implémentation de 3 algos et comparaisons des résultats*************

# Implémentation de l'algorithme naive-Bayes

GaussNb = GaussianNB() #appel de l'algorihtme naive-bayes
GaussNb.fit(X_train, Y_train) #entrainer l'algorithme sur le set d'apprentissage
pred1= GaussNb.predict(X_test) # faire des prédictions sur le set de test
pred12=GaussNb.predict(X_train)

test_score00=accuracy_score(Y_test, pred1, normalize = True) #calcul du paramétre accuracy score
train_score01=accuracy_score(Y_train,pred12) #accuracy score sur le set d'apprentissage

pre1=precision_score(Y_test,pred1,average=None).mean() #calcul du score de précision
rec1=recall_score(Y_test,pred1,average=None).mean() #calcul du score de rappel

pre12=precision_score(Y_train,pred12,average=None).mean() #calcul du score de précision
rec12=recall_score(Y_train,pred12,average=None).mean() #calcul du score de rappel


#Affichage des résultats:

print("\n**********************************************\n")
print("**********Naive_Bayes algorithme**********\n")
print(" taux de reconnaissance pour la base de test :",test_score00,"\n")
print("***************************")
print(" taux de reconnaissance pour la base d'apprentissage :",train_score01,"\n")
print("***************************")
print ("score de précision pour la base de test :",pre1,"\n")
print ("score de précision pour la base d'appentissage :",pre12,"\n")
print("***************************")
print ("score de rappel pour la base de test: ",rec1,"\n")
print ("score de rappel pour la base d'apprentissage: ",rec12,"\n")
print("**************************\n")

table =pd.crosstab(Y_test,pred1)
#print(table)
plt.matshow(table)
plt.title("Matrice de Confusion test- naive \n")
plt.colorbar()
plt.show()

table2 =pd.crosstab(Y_train,pred12)
#print(table)
plt.matshow(table2)
plt.title("Matrice de Confusion train- naive \n")
plt.colorbar()
plt.show()

matrice_test=confusion_matrix(Y_test,pred1) #Matrice de confusion en utilisant les données de test
matrice_train=confusion_matrix(Y_train,pred12) #Matrice de confusion en utilisant les données d'apprentissage
print("**************************\n")
print("matrice de confusion (test):\n",matrice_test,"\n")
print("**************************\n")
print("matrice de confusion (appentissage):\n",matrice_train,"\n")
print("**************************\n")



#Implémentation de KNeighbors *************************************************

Kneigh = KNeighborsClassifier(n_neighbors=5) #appel de l'algorithme K neighbors avec N=5

Kneigh.fit(X_train, Y_train) # former le model à notre set d'appprentissage 

pred21 = Kneigh.predict(X_test) # faire des prédictions sur le set de test

test_score03 =accuracy_score(Y_test,pred21) #calcul de l'accuracy score
pre21=precision_score(Y_test,pred21,average=None).mean()#calcul du score de précision
rec21=recall_score(Y_test,pred21,average=None).mean()#calcul du score de rappel

pred22 = Kneigh.predict(X_train)

train_score03 =accuracy_score(Y_train,pred22) #calcul de l'accuracy score
pre22=precision_score(Y_train,pred22,average=None).mean()#calcul du score de précision
rec22=recall_score(Y_train,pred22,average=None).mean()#calcul du score de rappel

#Affichage des résultats :

print("\n************KNeighbors algorithme****************** \n")
print ("taux de reconnaissance pour le set de test: \n",test_score03,"\n")
print ("taux de reconnaissance pour le set d'appentrissage: ",train_score03,"\n")
print("*************************\n")
print ("score de précision pour le set de test \n:",pre21,"\n")
print ("score de précision pour le set d'apprentissage :",pre22,"\n")
print("*************************\n")
print ("score de rappel pour le set de test:",rec21,"\n")
print ("score de rappel pour le set d'apprentissage: ",rec22,"\n")
print("**************************\n")

table =pd.crosstab(Y_test,pred21)
#print(table)
plt.matshow(table)
plt.title("Matrice de Confusion test- KNeighbors \n")
plt.colorbar()
plt.show()


table2 =pd.crosstab(Y_train,pred22)
#print(table)
plt.matshow(table2)
plt.title("Matrice de Confusion train- KNeighbors \n")
plt.colorbar()
plt.show()

matrice_test=confusion_matrix(Y_test,pred21) #Matrice de confusion en utilisant les données de test
matrice_train=confusion_matrix(Y_train,pred22) #Matrice de confusion en utilisant les données d'apprentissage

print("**************************\n")
print("matrice de confusion (test):\n",matrice_test,"\n")
print("**************************\n")
print("matrice de confusion (appentissage):\n",matrice_train,"\n")
print("**************************\n")

#Apprentissage avec l'arbre decision tree *************************************

dtree = DecisionTreeClassifier() #appel du model decisionTree
dtree.fit(X_train, Y_train) #le model apprend sur la base d'apprentissage

X_test_pred=dtree.predict(X_test) #prediction sur test set
X_train_pred=dtree.predict(X_train)#prediction sur le train set

test_score=accuracy_score(Y_test,X_test_pred) #accuracy score sur le test set
train_score=accuracy_score(Y_train,X_train_pred) #accuracy score sur le set d'apprentissage

pre3=precision_score(Y_test,X_test_pred,average=None).mean()#score de précision sur le set de test
rec3=recall_score(Y_test,X_test_pred,average=None).mean()#score de rappel sur le set de test

pre4=precision_score(Y_train,X_train_pred,average=None).mean()#score de précision sur le set d'apprentissage
rec4=recall_score(Y_train,X_train_pred,average=None).mean()#score de rappel sur le set d'appentissage

#Affichage des resultats
print("************Decision Tree algorithme ********** \n")
print ("accuracy_score of the test set:",test_score,"\n")
print("**************************\n")
print ("accuracy_score of the train set :",train_score,"\n")
print("**************************\n")
##******************************************************************************

print ("precison score of the test set :",pre3,"\n")
print ("precison score of the train set :",pre4,"\n")
print("**************************\n")
print ("recall score of the test set:",rec3,"\n")
print ("recall score of the train set :",rec4,"\n")
print("**************************\n")

#Algorithme de Validation croisée 
print("************Cross Val algorithme ********** \n")
Scores = cross_val_score(dtree, X_train, Y_train, scoring=None, cv=4)
print('cross val scores are :',Scores)
print("**************************\n")
print("la moyennes des scores obtenus par Cross_vall est ",Scores.mean())
print("**************************\n")

#Affichage des données graphiquement et analytiquement de façon matricielle

table =pd.crosstab(Y_test,X_test_pred)
#print(table)
plt.matshow(table)
plt.title("Matrice de Confusion test- DT \n")
plt.colorbar()
plt.show()

table2 =pd.crosstab(Y_train,X_train_pred)
plt.matshow(table2)
plt.title("Matrice de Confusion train- DT \n")
plt.colorbar()
plt.show()

#la matrice de confusion sur les données d’apprentissage ainsi que les données de test
matrice_test=confusion_matrix(Y_test,X_test_pred) #Matrice de confusion en utilisant les données de test

matrice_train=confusion_matrix(Y_train,X_train_pred) #Matrice de confusion en utilisant les données d'apprentissage

print("**************************\n")
print("matrice de confusion (test):\n",matrice_test,"\n")
print("**************************\n")
print("matrice de confusion (appentissage):\n",matrice_train,"\n")
print("**************************\n")

#Reconstuire le modéle en variant les hyperparamétres du model DecisionTree
#On varie la profondeur maximal de l'arbre 

i=0
for i in range(5):
    i=i+1
    dtree = DecisionTreeClassifier(max_depth=i) #en ajoute l'hyperparamétre de profondeur
    dtree.fit(X_train, Y_train) 
    X_test_pred=dtree.predict(X_test)
    X_train_pred=dtree.predict(X_train)
    test_score=accuracy_score(Y_test,X_test_pred)
    train_score=accuracy_score(Y_train,X_train_pred)
    print ("taux de reconnaissance du set de test pour une pronfondeur max de (",i,") est :",test_score,"\n")
    print ("taux de reconnaissance du set d'apprentissage pour une profondeur max de (",i,") est :",train_score,"\n")
    print("**************************\n")


# Crée un model avec de nouveaux sous-ensemble 
    #taille moyenne des paquets, min,max et le temps inter-arrivé entre les paquets.....etc 
    
#Exemple ===================>ss_ensemble=4 on aura le ss ensemble size/average/min/max seulement 
    ###les 4 premiers attributs seuelemnt donc 
    
def new_model(ss_ensemble):
    
    model_bis=data[list(data.columns[:ss_ensemble])] #les sous_ensembles sont pris de façon direct selon le fichier csv
    X2_train,X2_test,Y2_train, Y2_test = train_test_split(model_bis,Y, test_size=0.3) #Séparer la data entre un set de test et d'apprentissage

    dtree = DecisionTreeClassifier() #en utilisant le model de DecisonTree
    dtree.fit(X2_train, Y2_train) # adapter le model pour le set d'apprentissage 
 
    predict_test_set=dtree.predict(X2_test) 
    predict_train_set=dtree.predict(X2_train)
    
    test_score2=accuracy_score(Y2_test,predict_test_set) 
    train_score2=accuracy_score(Y2_train,predict_train_set) 
    
    print("le taux de reconnaissance sur l'ensemble de test du nouveau model\n:",test_score2,'\n')
    print("le taux de reconnaissance sur l'ensemble d'apprentissage du nouveau model \n:",train_score2,'\n')

ss_ensemble=int(input("entrer le nombre de sous ensembles dont vous voulez faire l'étude : \n "))
new_model(ss_ensemble)

#entrainer un modèle que sur les flux DHCP des données d'apprentissage*********

Re=data[data['DHCP']==1] #un modèle entraîné que sur DHCP
#print(Re)
Y_dhcp = Re["type"] #on ajoute les étiquettes
#print("**************************\n",Y_dhcp)

X_dhcp = Re[list(data.columns[:16])]
#print(X_dhcp)

X_dhcp_train,X_dhcp_test,Y_dhcp_train, Y_dhcp_test = train_test_split(X_dhcp,Y_dhcp, test_size=0.3)
#Séparer les données en set de test et d'apprentissage

dtree = DecisionTreeClassifier()
dtree.fit(X_dhcp_train, Y_dhcp_train) #ajuster le model decisionTree au données d'apprentissage

pred_dhcp=dtree.predict(X_dhcp_test)  #faire des prédictions sur les données de test 
test_score3=accuracy_score(Y_dhcp_test,pred_dhcp) #calcul du taux de reconnaisance du model au set de données DHCP test

pred_dhcp2=dtree.predict(X_dhcp_train) #faire des prédictions sur les données de d'apprentissage
train_score3=accuracy_score(Y_dhcp_train,pred_dhcp2)#calcul du taux de reconnaisance du model au set de données DHCP train
#Affichage du résultat
print("taux de reconnaissance sur base de test pour l'attribut DHCP uniquement  : ",test_score3)   
print("taux de reconnaissance sur base d'appentissage pour l'attribut DHCP uniquement  : ",train_score3)
print("**************************\n")



##***************************************************** Partie 02**************************************
#******************************************************************************************************

#Combiner les attributs textuelles aux attributs numériques et refaire l'étude 
#data_text= pd.read_csv('datasetTextual.csv') #lire les données du fichier des attrbibuts textuelles
#data_merge=data.merge(data_text,how='outer') #combiner le vecteur des attributs textuelles

print("****************************2éme Partie *********************")
print("**********************données textuelles & numériques ********")


data_merge= pd.read_csv('dataset.csv') #lire les données du fichier des attributs numériques et textuelles
Y = data_merge["type"] #étiquette

X = data_merge.drop(labels="type", axis=1) #enlever la coonne de type

X_train,X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.3)#séparer les données en train & test

#Implémentation des algos  issues de la partie 1 pour faire l'étude de la 
    #partie 2 ou les bases de données sont combinées
"""
# Implémentation de l'algorithme naive-Bayes***********************************

GaussNb = GaussianNB() #appel de l'algorihtme naive-bayes
GaussNb.fit(X_train, Y_train) #entrainer l'algorithme sur le set d'apprentissage
pred1= GaussNb.predict(X_test) # faire des prédictions sur le set de test
pred12=GaussNb.predict(X_train)

test_score00=accuracy_score(Y_test, pred1, normalize = True) #calcul du paramétre accuracy score
train_score01=accuracy_score(Y_train,pred12) #accuracy score sur le set d'apprentissage

pre1=precision_score(Y_test,pred1,average=None).mean() #calcul du score de précision
rec1=recall_score(Y_test,pred1,average=None).mean() #calcul du score de rappel

pre12=precision_score(Y_train,pred12,average=None).mean() #calcul du score de précision
rec12=recall_score(Y_train,pred12,average=None).mean() #calcul du score de rappel


#Affichage des résultats
print("\n**********************************************\n")
print("**********Naive_Bayes algorithme**********\n")
print(" taux de reconnaissance pour la base de test :",test_score00,"\n")
print("***************************")
print(" taux de reconnaissance pour la base d'apprentissage :",train_score01,"\n")
print("***************************")
print ("score de précision pour la base de test :",pre1,"\n")
print ("score de précision pour la base d'appentissage :",pre12,"\n")
print("***************************")
print ("score de rappel pour la base de test: ",rec1,"\n")
print ("score de rappel pour la base d'apprentissage: ",rec12,"\n")
print("**************************\n")


#implémentation de l'algorithme Kneighbors*************************************
Kneigh = KNeighborsClassifier(n_neighbors=5) #appel de l'algorithme K neighbors avec N=5
Kneigh.fit(X_train, Y_train) # former le model à notre set d'appprentissage 

pred21 = Kneigh.predict(X_test) # faire des prédictions sur le set de test
test_score03 =accuracy_score(Y_test,pred21) #calcul de l'accuracy score

pre21=precision_score(Y_test,pred21,average=None).mean()#calcul du score de précision
rec21=recall_score(Y_test,pred21,average=None).mean()#calcul du score de rappel

pred22 = Kneigh.predict(X_train)

train_score03 =accuracy_score(Y_train,pred22) #calcul de l'accuracy score
pre22=precision_score(Y_train,pred22,average=None).mean()#calcul du score de précision
rec22=recall_score(Y_train,pred22,average=None).mean()#calcul du score de rappel
#Affichage des résultats
print("\n************KNeighbors algorithme****************** \n")
print ("taux de reconnaissance pour le set de test: \n",test_score03,"\n")
print ("taux de reconnaissance pour le set d'appentrissage: ",train_score03,"\n")
print("*************************\n")
print ("score de précision pour le set de test \n:",pre21,"\n")
print ("score de précision pour le set d'apprentissage :",pre22,"\n")
print("*************************\n")
print ("score de rappel pour le set de test:",rec21,"\n")
print ("score de rappel pour le set d'apprentissage: ",rec22,"\n")
print("**************************\n")
"""
#Implémentation de DecisionTree pour les données combinées ********************

dtree = DecisionTreeClassifier() #appel du model DecisionTree
dtree.fit(X_train,Y_train) #adapter le model au set d'apprentissage 

X_test_pred=dtree.predict(X_test) #prediction sur le set de test
X_train_pred=dtree.predict(X_train) #prediction sur le set d'apprentissage

test_score=accuracy_score(Y_test,X_test_pred) #calcul du taux de reconnaissance sur le set de test
train_score=accuracy_score(Y_train,X_train_pred)#calcul du taux de reconnaissance sur le set de d'apprentissage

pre5=precision_score(Y_test,X_test_pred,average=None).mean()
pre6=precision_score(Y_train,X_train_pred,average=None).mean()

rec5=recall_score(Y_test,X_test_pred,average=None).mean()
rec6=recall_score(Y_train,X_train_pred,average=None).mean()

#Affichage des résusltats :

print ("taux de reconnaissance du test set:",test_score)
print("**************************\n")
print ("taux de reconnaissance du train set :",train_score)
print("**************************\n")
print ("precison score du test set :",pre5,"\n")
print("**************************\n")
print ("precison score du train set :",pre6,"\n")
print("**************************\n")
print ("recall score du test set:",rec5,"\n")
print("**************************\n")
print ("recall score du train set  :",rec6,"\n")
print("**************************\n")

table =pd.crosstab(Y_test,X_test_pred)
#print(table)
plt.matshow(table)
plt.title("Matrice de Confusion test- merged \n")
plt.colorbar()
plt.show()

table2 =pd.crosstab(Y_train,X_train_pred)
#print(table)
plt.matshow(table2)
plt.title("Matrice de Confusion train- merged \n")
plt.colorbar()
plt.show()

matrice_test=confusion_matrix(Y_test,X_test_pred) #Matrice de confusion en utilisant les données de test
matrice_train=confusion_matrix(Y_train,X_train_pred) #Matrice de confusion en utilisant les données d'apprentissage
print("**************************\n")
print("matrice de confusion (test):\n",matrice_test,"\n")
print("**************************\n")
print("matrice de confusion (appentissage):\n",matrice_train,"\n")
print("**************************\n")

