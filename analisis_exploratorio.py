#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grupo: NAN
Integrantes: Rozas Chavez, Antuanette Carolina
             Madril, Joaquin Leandro
             Fernández Fazio, Adrián Patricio
             
 TP 2
             
Este archivo incluye todo el código para realizar el análisis exploratorio

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

#%% Cargamos el csv

letras_df = pd.read_csv('TP02-EnglishTypeAlphabet.csv')

#%% Ejercicio 1, análisis exploratorio

n_filas, n_columnas = letras_df.shape

print(f"Cantidad total de imágenes: {n_filas}")
print(f"Cantidad total de atributos: {n_columnas}")

""" Constatamos que para cada letra hay 1016 variantes. La clase está dada por
la etiqueta label, hay 26416 filas y 785 atributos(incluyendo label)"""
letras = "abcdefghijklmnopqrstuvwxyz"

for i in range(26):
    cantidad_de_variantes = (letras_df['label'] == i).sum()
    print(f"La clase {letras[i]} ({i}) tiene {cantidad_de_variantes} variantes")

#%% Graficamos las letras

primera_variante_por_letra = {}

j = 0

#Este loop popula el diccionario con la primera variante de cada letra
for i in range(0,26416,1016):
    primera_variante_por_letra[letras[j]] = i
    j += 1

letras_df_sin_label = letras_df.drop(['label'], axis=1) #Necesario para reshape 
#Graficamos 5 variantes por letra al azar
np.random.seed(2)
for i in range(0, 26416, 1016):
    # 1. Creamos la figura con 5 subplots (1 fila, 5 columnas)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    lista_aleatoria = np.random.randint(i, i + 1016, size=5)
    
    for idx_plot, j in enumerate(lista_aleatoria):
        img = np.array(letras_df_sin_label.iloc[j]).reshape((28,28))
        
        # Dibujamos en el subplot correspondiente
        axes[idx_plot].imshow(img, cmap='gray')
        axes[idx_plot].set_title(f"Imagen {j}") 
        
    plt.show() # Muestra el gráfico con las 5 letras juntas
    

#%% 
"""Graficamos una letra de cada clase para pensar qué atributos corresponden 
al 'marco' de pixeles irrelevantes"""
for i in range(0, 26416, 1016):
    # fig = plt.subplots(1, 1, figsize=(8, 8))
    img = np.array(letras_df_sin_label.iloc[i]).reshape((28,28))
    plt.imshow(img, cmap='gray')
    plt.show()
#%% Creamos un dataset con las O y las L
letras_OL_df = letras_df[(letras_df['label'] == 14) | (letras_df['label'] == 11)]
letras_OL_df.reset_index(inplace=True, drop=True)

#%% Separamos train y test
# 'X' son todas las columnas menos 'label'
X = letras_OL_df.drop(columns=['label'])

# 'y' es solo la columna 'label'
y = letras_OL_df['label']
# Dividimos los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,       
    random_state=2,     
    stratify=y          
)
#%% Creamos mapas de calor para ambas letras
umbral_negro = 170

# Buscamos las O y L's 'promedio'
letras_agrupadas_sum = letras_OL_df.groupby('label').sum() // 1016

img = np.array(letras_agrupadas_sum.iloc[0]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.grid()
plt.show()
           
img = np.array(letras_agrupadas_sum.iloc[1]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.grid()
plt.show()

#%% Elegimos 3 atributos
def evaluar_y_mostrar_knn(lista_de_atributos, k):
    columnas = ['pixel ' + str(atributo) for atributo in lista_de_atributos]
    clasificador = KNeighborsClassifier(n_neighbors=k)
    clasificador.fit(X_train[columnas].values, y_train.values)
    y_pred = clasificador.predict(X_test[columnas].values)
    exactitud = accuracy_score(y_test.values, y_pred)
    matriz = confusion_matrix(y_test.values, y_pred)
    print('*'*25,
        f"knn con pixeles {lista_de_atributos} y k={k}",
        f"Exactitud: {round(exactitud, 3)}",
        "matriz de confusion:",
        matriz,
        '_'*25,
        sep='\n')
    return (exactitud, matriz)

#%% Probamos casos
_ = evaluar_y_mostrar_knn([0, 1, 2], 5) #caso de control


# ELegimos 10(filas) * 28(columnas) + 21(columna específica, elegida por mapa) 
_ = evaluar_y_mostrar_knn([301, 302, 303], 5) #Elegido por mapa

_ = evaluar_y_mostrar_knn([299, 300, 301], 5) #Desplazado a la izq.

_ = evaluar_y_mostrar_knn([299-28, 300-28, 301-28], 5) #line up

_ = evaluar_y_mostrar_knn([299+28, 300+28, 301+28], 5) #line down

_ = evaluar_y_mostrar_knn([299+56, 300+56, 301+56], 5) # 2 line down

#%%
"""Como detectamos que los atributos 299, 300, 301 eran la mejor elección 
según las métricas, probamos ampliando la cantidad de atributos a 7"""
_ = evaluar_y_mostrar_knn([i for i in range(297,304)], 5)

# Mejoraron las métricas, ampliamos atributos a 11
_ = evaluar_y_mostrar_knn([i for i in range(295,306)], 5)

# Mejoraron las métricas, ampliamos atributos a 15
_ = evaluar_y_mostrar_knn([i for i in range(293,308)], 5) # Detecta el 100% de las 'L'


#%% Probamos distintos K's
def evaluar_distintos_k(pixeles):
    lista_k = []
    lista_exactitud = []
    
    for k in range(1, 51, 3):
        exactitud, matriz = evaluar_y_mostrar_knn(pixeles, k) 
        lista_k.append(k)
        lista_exactitud.append(exactitud)
        
    plt.figure(figsize=(10, 6))
    plt.plot(lista_k, lista_exactitud, marker='o')
    plt.title('Exactitud del modelo KNN para distintos valores de K')
    plt.xlabel('K')
    plt.ylabel('Exactitud (Accuracy)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

#%%
evaluar_distintos_k([299, 300, 301]) #mejor conjunto de 3, distinto k

#%% Probamos distintos K's y más atributos
evaluar_distintos_k([i for i in range(293,308)]) #los 15 atributos de antes


""" Observamos que con K=1 la exactitud alcanzada es muy alta, 0,995"""

#=========================================================
#%% clasificacion multiclase
#separamos datos en held-out y dev

# 'X' son todas las columnas menos 'label'
X = letras_df.drop(columns=['label'])

# 'y' es solo la columna 'label'
y = letras_df['label']
# Dividimos los datos
X_dev, X_held_out, y_dev, y_held_out = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2,
    stratify=y
)
#%% separamos dev en train y test para esta seccion

X_train, X_test, y_train, y_test = train_test_split(
    X_dev,
    y_dev,
    test_size=0.2,
    random_state=2,
    stratify=y_dev
)

arbol = DecisionTreeClassifier(max_depth=10)
arbol.fit(X_train.values, y_train.values)

y_pred = arbol.predict(X_test.values)
print(accuracy_score(y_test.values, y_pred))
matriz = confusion_matrix(y_test, y_pred)
#plt.figure(figsize=(20,10))
#plot_tree(arbol,
#          max_depth=2,
#          feature_names=X_train.columns,
#          class_names=list(letras),
#          rounded=True,
#          filled=True
#          )
#plt.show()
#%% 
precision = []
profundidad = []

for d in range(1, 21, 2):
    arbol = DecisionTreeClassifier(max_depth=d)
    arbol.fit(X_train.values, y_train.values)
    y_pred = arbol.predict(X_test.values)
    precision.append(accuracy_score(y_test.values, y_pred))
    profundidad.append(d)

plt.plot(profundidad, precision)
plt.grid()
plt.show()


#%%

#revisar
#prepararse un cafe durante la ejecucion

arboles_precision = []
for atributos in range(1, 202, 10):
    
    for altura in range(1, 11, 2):
        arbol = DecisionTreeClassifier(max_depth=altura, max_features=atributos)
        arbol.fit(X_train.values, y_train.values)
        y_pred = arbol.predict(X_test.values)
        precision = accuracy_score(y_test.values, y_pred)
        
        arboles_precision.append([atributos, altura, precision])

arboles_precision_df = pd.DataFrame(np.array(arboles_precision),
                                    columns=["cant_atributos",
                                             "altura_arbol",
                                             "precision"]
                                    )
#solo el maximo de cada uno
maximos = arboles_precision_df.groupby("cant_atributos")["precision"].idxmax()
arboles_precision_df = arboles_precision_df.loc[maximos]

plt.plot(arboles_precision_df["cant_atributos"],
         arboles_precision_df["precision"]
    )
plt.grid()
plt.show()

#%%
#habia que hacerlo con k folding
#esta linea hace arbol.fit haciendo k folding (predeterminado es 5)
#devuelve una lista con la exactitud para cada grupo
cross_val_score(arbol, X_dev, y=y_dev, scoring=accuracy_score)





















