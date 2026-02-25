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

#Este loop popula el diccionario con la primera variante de cada letra
for i in range(0,26416,1016):
    primera_variante_por_letra[letras[j]] = i
    j += 1

letras_df_sin_label = letras_df.drop(['label'], axis=1) #Necesario para reshape 
#Graficamos 5 variantes por letra al azar
for i in range(0, 26416, 1016):
    # 1. Creamos la figura con 5 subplots (1 fila, 5 columnas)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    lista_aleatoria = np.random.randint(i, i + 1016, size=5)
    
    for idx_plot, j in enumerate(lista_aleatoria):
        img = np.array(letras_df_sin_label.iloc[j]).reshape((28,28))
        
        # Dibujamos en el subplot correspondiente
        axes[idx_plot].imshow(img, cmap='gray')
        
    plt.show() # Muestra el gráfico con las 5 letras juntas
    

#%% 
"""Graficamos una letra de cada clase para pensar qué atributos corresponden 
al 'marco' de pixeles irrelevantes"""
for i in range(0, 26416, 1016):
    # fig = plt.subplots(1, 1, figsize=(8, 8))
    img = np.array(letras_df_sin_label.iloc[i]).reshape((28,28))
    plt.imshow(img, cmap='gray')
    plt.show()

    