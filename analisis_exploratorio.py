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

letras_df = pd.read_csv('TP02-EnglishTypeAlphabet.csv')

n_filas, n_columnas = letras_df.shape

print(f"Cantidad total de imágenes: {n_filas}")
print(f"Cantidad total de atributos: {n_columnas}")