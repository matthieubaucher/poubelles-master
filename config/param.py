# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:06:30 2021

@author: 33675
"""

chemin_appli = "C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application"
#chemin_video = "C:/Users/33675/talan.com/GRP-Veolia - Expérimentation vidéo - General/02-Banque vidéos/"
chemin_video = "C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/video_test/"

# Chargement des poids du modèle RCNN
poids_rcnn = "../weights/mask_rcnn_sacv_tlb_3c_v2_res101_20epoch_lr0005_1093.h5"

# Configuration du smtp
smtp_adress = 'smtp.gmail.com'
smtp_port = 465

# Configuration de l'expéditeur
sender_email = '@gmail.com'
sender_name = 'Talan'
password = ''

receiver_emails = '@gmail.com'
receiver_names = 'Tanguy'
