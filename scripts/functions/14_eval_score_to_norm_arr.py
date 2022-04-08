def ImgScoreListNormalized(path, model):
    """

    This function performs a weighted-score array normalization from a pic dir

    Input:
    path            : img path
    model           : Weights file

    Output:
    obj_arr         : Objects detection 
    img_name_arr    : Images names
    score_arr       : Pictures scores 
    normalized_arr  : Pictures scores normalized (numpy array)
    
    """
    
    path_image      = path
    img_name_arr    = []
    score_arr       = []
    obj_arr         = []
    threshold       = 100

    for image in listdir(path_image):
        
        # Add img name to list
        img_name_arr.append(image)
        
        # Apply model on img
        image_test      = load_img(path + image)
        image_test      = img_to_array(image_test)
        result          = model.detect([image_test])
        r               = result[0]
        
        # Numbering classes
        nb_sac          = (r['class_ids'] == 1).sum()
        nb_cart         = (r['class_ids'] == 2).sum()
        nb_bout         = (r['class_ids'] == 3).sum()
        
        # Weighted Score calculation
        img_score       = 5*nb_sac + 3*nb_cart + nb_bout
        # Objets detected sum
        # nb_obj = nb_sac + nb_cart+ nb_bout
        nb_obj          = len(r['class_ids'])
        
        # Limit high score by 
        if img_score > threshold:
            img_score   = threshold
        # Add score to list
        score_arr.append(img_score)
        # Add obj to list
        obj_arr.append(nb_obj)

    normalized_arr      = preprocessing.normalize([score_arr])
    return obj_arr, score_arr, normalized_arr, img_name_arr