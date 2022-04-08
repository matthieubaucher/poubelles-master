# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 10:14:58 2021

@author: 33675
"""

# Lancement du modèle de détection d'objet sur une image et enregistrement des prédictions
def savePredImg(model, image, cpt):
    
    image_array = img_to_array(image)
    result = model.detect([image_array])
    r = result[0]
    display_instances2(
        image_array, r['rois'], r['masks'], r['class_ids'],
        ["0", "sac", "cart", "bout"], r['scores'],
        title = "Prédictions"
    )
    