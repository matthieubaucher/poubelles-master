# Affichage des images d'un répertoire, avec la prédiction du modèle (libellé, zone, score de prédiction)   
def ImgScoring(image, model):
    """
    
    This function brings infos from a vectorized picture

    Input: 
    image               : arr (vectorized picture) 
    model               : weights_file

    Output:
    img_score           : Weighted score
    obj_sum             : Object numbering in image
    obj_size_perc_arr   : Storing obj surfaces in arr
    
    """   

    # Apply model on img
        # image_test  = load_img(image)
        # image_test  = img_to_array(image_test)
    image_test          = image
    result              = model.detect([image_test])
    r                   = result[0]

    # Numbering classes
    nb_sac              = (r['class_ids'] == 1).sum()
    nb_cart             = (r['class_ids'] == 2).sum()
    nb_bout             = (r['class_ids'] == 3).sum()
    
    # Weighted Score calculation
    obj_sum             = len(r['class_ids'])
    img_score           = 5*nb_sac + 3*nb_cart + nb_bout
    obj_size_perc_arr   = []

    # Storing all objects surfaces in arr
    for line in range(obj_sum):
        nparr                           = r['rois'][line]
        obj_array                       = (nparr[3]-nparr[1])*(nparr[2]-nparr[0])
        img_array                       = image_test.shape[0]*image_test.shape[1]
        obj_size_vs_img_size_percent    = (obj_array/img_array)*100
        obj_size_perc_arr.append(obj_size_vs_img_size_percent)

    return img_score, obj_sum, obj_size_perc_arr