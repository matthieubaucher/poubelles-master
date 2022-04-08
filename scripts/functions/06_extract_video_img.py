# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:02:37 2021

@author: 33675
"""

# Extrait une image à partir d'une vidéo, si elle existe selon un pas donné
def extractVideoImg(sec):
    vid.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames,image = vid.read()
    if hasFrames:
        cv2.imwrite(chemin_appli + '/data/demonstration/dechet_detect/predicted_img.png', image)
    return (hasFrames, image)