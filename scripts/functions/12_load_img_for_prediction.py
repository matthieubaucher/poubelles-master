# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 15:11:41 2021

@author: 33675
"""

# Importer une image et la mettre dans un format conforme à celui du modèle qui fait la prédiction (format utilisé pour l'entraînement du modèle)
def loadImgForPr(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32') / 255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis = 0)
   return np_image