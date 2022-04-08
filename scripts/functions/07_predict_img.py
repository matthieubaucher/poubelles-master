# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:04:47 2021

@author: 33675
"""

# Prédiction de la présence de déchet sur une image, via le modèle de classification
def predictClassif(cpt, model):
    
    batch_size = 16
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        chemin_appli + '/data/demonstration/dechet_detect/',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False )
    
    y_pred = model.predict(test_generator) 
    y_pred = (y_pred > 0.01) * 1
    y_pred = y_pred.ravel()
    
    if y_pred == 1:
        cpt = cpt + 1
        chdir(chemin_appli)
    else: 
        cpt = 0
        os.remove(chemin_appli + '/data/demonstration/dechet_detect/img/predicted_img.png')
    return(cpt)

# Prédiction de la présence de déchet sur une image, via le modèle de classification
def predictClassif2(image, cpt, model):

    imgv1 = img_to_array(image).astype('float32')
    imgv2 = cv2.resize(imgv1, (224,224))
    imgv3 = np.expand_dims(imgv2, axis=0)
    
    y_pred = model.predict(imgv3)
    y_pred = (y_pred > 0.1) * 1
    
    if y_pred == 0:
        cpt = cpt + 1
        chdir(chemin_appli)
        cv2.imwrite('data/demonstration/dechet_detect/predicted_img.png', image)
    else: 
        cpt = 0
        
    return(cpt)

# Prédiction de la présence de déchet sur une image, via le modèle de détection d'objet
def predictDetect(image, cpt, model):
    img = img_to_array(image)
    result = model.detect([img])
    r = result[0]
    sc = r["scores"]
    if sc.size >= 1 :
        cpt = cpt + 1
    else :
        cpt = 0
    return(cpt)