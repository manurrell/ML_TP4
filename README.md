# 🧠 Unsupervised Learning & Generative Models Toolkit

![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)

## 📋 Descripción del Proyecto

Este repositorio contiene una implementación modular y exhaustiva de algoritmos fundamentales de **Aprendizaje No Supervisado** y **Deep Learning Generativo**.

El objetivo principal del proyecto es resolver problemas de **agrupamiento de datos (clustering)**, **reducción de dimensionalidad** y **generación de contenido sintético**. A diferencia de utilizar implementaciones de caja negra, este proyecto incluye el desarrollo desde cero ("from scratch") o implementaciones personalizadas en PyTorch de la lógica central de los algoritmos.

El sistema es capaz de:
1.  Identificar patrones latentes en datos no etiquetados mediante técnicas de Clustering.
2.  Comprimir información visual compleja (FashionMNIST) preservando la varianza.
3.  Generar nuevas muestras de imágenes de ropa utilizando un Autoencoder Variacional (VAE).

## 🚀 Tecnologías Utilizadas

El proyecto ha sido construido utilizando un stack científico robusto:

* **Core Logic:** `Python 3`
* **Deep Learning:** `PyTorch` (para la arquitectura del VAE y optimización por GPU).
* **Cálculo Numérico:** `NumPy` y `SciPy`.
* **Visualización:** `Matplotlib` (para graficar clusters, curvas de pérdida y espacio latente).
* **Entorno de Desarrollo:** `Jupyter Notebooks`.

## ⚙️ Arquitectura y Módulos 


| Módulo | Algoritmo | Descripción |
| :--- | :--- | :--- |
| **K_means** | K-Means Clustering | Implementación iterativa para partición de datos basada en centroides. |
| **GMM** | Gaussian Mixture Models | Modelo probabilístico que asume que los datos provienen de una mezcla de distribuciones gaussianas finitas. |
| **DBSCAN** | Density-Based Clustering | Algoritmo de agrupamiento basado en densidad para encontrar clusters de forma arbitraria y ruido. |
| **PCA** | Principal Component Analysis | Técnica de reducción de dimensionalidad lineal para proyección de features. |
| **VAE** | Variational Autoencoder | Red neuronal (Encoder/Decoder) implementada en `torch.nn` para aprender representaciones latentes y generar nuevas imágenes. |

## 📊 Resultados Destacados

El proyecto demuestra la eficacia de los modelos a través de:

* **Análisis de Clusters:** Comparación visual y métrica (método del codo) entre K-Means, GMM y DBSCAN sobre datasets 2D.
* **Espacio Latente:** Visualización de la compresión de imágenes del dataset **FashionMNIST**.
* **Generación de Imágenes:** Muestreo del espacio latente del VAE para crear nuevas prendas de ropa sintéticas que no existen en el dataset original.
