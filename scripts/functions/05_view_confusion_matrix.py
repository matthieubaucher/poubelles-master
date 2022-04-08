# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:42:44 2021

@author: 33675
"""

# Fonction pour avoir le tableau des predictions ainsi que matrice de confusion + classification_report de sklearn
def confusionMatrix(path,model_detect,conf):
    score = 0
    path_image = path
    tab = pd.DataFrame(columns=['image','y', 'y_pred','score'])
    
    for image in listdir(path_image):
        
        temp = image[1]
        if temp == "c":
            y = "clean"
        else :
            y = "dirty"
            
        image_test = load_img(path + image)
        image_test = img_to_array(image_test)
        result = model_detect.detect([image_test])
        r = result[0]
        sc = r["scores"]
        
        if sc.size == 0:
            score = sc
        elif sc.size >= 1:
            score = max(sc)
            
        if score >= conf :
            y_pred = "dirty"
        else :
            y_pred = "clean"
            
        print("y_pred => ")
        print(y_pred)
        print("r => ")
        print(r)
        print("sc =>")
        print(sc)
            
        tab = tab.append({'image':image,'y': y, 'y_pred': y_pred,'score':score}, ignore_index=True)
        
    cm = confusion_matrix(tab["y"], tab["y_pred"], labels = ["dirty","clean"])
    
    cr = classification_report(tab["y"], tab["y_pred"], target_names=["dirty","clean"])
    return(cm, cr,tab)