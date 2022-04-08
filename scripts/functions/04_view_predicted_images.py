# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:22:47 2021

@author: 33675
"""

# Affichage des images d'un répertoire, avec la prédiction du modèle (libellé, zone, score de prédiction)   
def viewPredImg(path, model_detect):
    
    path_image = path
    
    for image in listdir(path_image):
        image_test = load_img(path + image)
        image_test = img_to_array(image_test)
        result = model_detect.detect([image_test])
        r = result[0]
        visualize.display_instances(image_test, r['rois'], r['masks'], r['class_ids'],                                    
                                    ["0", "sac", "cart", "bout"], r['scores'],
                                    title = "Predictions")