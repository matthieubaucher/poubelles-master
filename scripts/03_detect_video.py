# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:58:10 2021

@author: 33675
"""

# Chemin vers la vidéo
list_of_files = glob.glob(chemin_video + "*")
#file_video = chemin_video + "GOPR0508.MP4"
file_video = max(list_of_files, key = os.path.getctime)

# Chargement de la vidép
vid = cv2.VideoCapture(file_video)

list_of_files = glob.glob(chemin_video + "*")
file_video = max(list_of_files, key = os.path.getctime)

print("File: ", file_video)

# Chargement de la vidéo
vid = cv2.VideoCapture(file_video)

# Envoie mail
sec = 0
intervalle = 5 # image toutes les x secondes
cpt = 0 # Compteur detection
test, image_vid = extractVideoImg(sec) # On récupère l'image si elle existe
image_vid = cv2.cvtColor(image_vid, cv2.COLOR_BGR2RGB)

# Values init
img_score = 0

while test :
    #pred, cpt = predictImg(image, cpt) # On prédit sur l'image_vid
    cpt = predictDetect(image_vid, cpt, model_detect) # On prédit sur l'image

    if cpt == 3 : # Si c'est la deuxième fois d'affilé qu'on observe un déchet alors envoie d'un mail
        img_score, obj_sum, obj_size_perc_arr = ImgScoring(image_vid, model_detect)
        print("***** img_score :", img_score)
        savePredImg(model_detect, image_vid, cpt)
        DetectBlurFace3()
        send_email(img_score)
        cpt = 0 # Une fois le mail envoyé on remet le compteur à 0
    sec = sec + intervalle #
    sec = round(sec, 2)
    test, image_vid = extractVideoImg(sec) # On récupère l'image si elle existe
    image_vid = cv2.cvtColor(image_vid, cv2.COLOR_BGR2RGB)
