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
import seaborn as sns
import numpy as np

letras_df = pd.read_csv('TP02-EnglishTypeAlphabet.csv')

#%% Ejercicio 1, análisis exploratorio

n_filas, n_columnas = letras_df.shape

print(f"Cantidad total de imágenes: {n_filas}")
print(f"Cantidad total de atributos: {n_columnas}")

""" Constatamos que para cada letra hay 1016 variantes. La clase está dada por
la etiqueta label, hay 26416 filas y 785 atributos(incluyendo label)"""

for i in range(26):
    cantidad_de_variantes = (letras_df['label'] == i).sum()
    print(f"La clase {i} tiene {cantidad_de_variantes} variantes")

#%% Graficamos las letras

primera_variante_por_letra = {}
letras = "abcdefghijklmnopqrstuvwxyz"

j = 0
for i in range(0,1016,1016):
    primera_variante_por_letra[letras[j]] = i
    j += 1

letras_df_sin_label = letras_df.drop(['label'], axis=1) 
img = np.array(letras_df_sin_label.iloc[12]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()

