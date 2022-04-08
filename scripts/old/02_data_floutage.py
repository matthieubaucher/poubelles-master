# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 10:30:10 2021

@author: 33675
"""

# Chargement de l'image orginale
src = cv2.imread('data/data_original/depot1.png', cv2.IMREAD_UNCHANGED)


# blur = cv2.blur(src,(25,25))
# blur = cv2.GaussianBlur(src,(2,2), cv2.BORDER_DEFAULT)
# median = cv2.medianBlur(src,5)
# blur = cv2.bilateralFilter(src,9,75,75)

# Fonction qui permet de faire de la data augmentation en ajoutant du flou
def dataFloutage(image, augment_facteur = 5, flou_facteur = 1):
   
    # Création et export des images
    i = 5
    while i < augment_facteur:
        i = i + 10
        blur = cv2.blur(image,(i*i, i*i))
        plt.subplot(122),plt.imshow(blur), plt.title('Blurred')
        fig.savefig('data/data_created3/full_figure.png')

  

dataFloutage(src, augment_facteur = 5, flou_facteur = 1)

blur = cv2.blur(src,(25,25))
blur = cv2.blur(src,(75,75))
blur = cv2.blur(src,(125,125))

plt.subplot(121),plt.imshow(src),plt.title('Original')
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')

