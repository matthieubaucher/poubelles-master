# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:08:32 2021

@author: tlebl
"""

# Fonction qui permet de faire de la data augmentation et l'export des images
def dataAugmentation(image, augment_facteur = 5, rotation = 0, luminosite = 0, couleurs = None, zoom = 0, pref_img = "img"):
    # print(pref_img)

    
    x = img_to_array(image) 
    # Reshape the input image 
    x = x.reshape((1, ) + x.shape) 
    datagen = ImageDataGenerator(rotation_range = rotation, channel_shift_range = luminosite, brightness_range = couleurs, zoom_range = zoom) 
    
    # Création et export des images
    i = 1
    for batch in datagen.flow(
            x, batch_size = 1, 
                      save_to_dir = 'data/data_test/',  
                      save_prefix = pref_img, save_format = 'jpg'
      ):
        i += 1
        if i > augment_facteur: 
            break
    
# Répertoire contenant les images
image_path = 'data/data_original'

# Liste des images
lst_img = [f for f in listdir(image_path) if isfile(join(image_path, f))]

# Parcours et augmentation des images (x10) + export
for f in enumerate(lst_img):
    img = load_img(image_path + "/" + f[1])
    dataAugmentation(img, augment_facteur = 10, rotation = 3, luminosite = 50, couleurs = [0.01,0.05], zoom = 0.1, pref_img = os.path.splitext(f[1])[0]) 


dataAugmentation(img, augment_facteur = 10, rotation = 3, couleurs = [0.01,0.05], zoom = 0.1, pref_img = os.path.splitext(f[1])[0]) 


channel_shift_range=150.0