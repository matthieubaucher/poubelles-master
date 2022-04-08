# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:52:08 2021

@author: 33675
"""

# Définir le répertoire qui contient les images et annotations
chemin_img = chemin_appli + "/data/entrainement"
chdir(chemin_appli + "/model")

# Chargement de la configuration du modèle
config = maskRcnnConfig()
config.display() # Vérifier que la configuration a bien été chargé

# Construction du dataset train
train_set = buildDataset()
train_set.buildDatasetLoading(chemin_img, is_train = True)
train_set.prepare()
print("Train: %d" % len(train_set.image_ids)) # Vérifier la taille du dataset


# Construction du dataset test
test_set = buildDataset()
test_set.buildDatasetLoading(chemin_img, is_train = False)
test_set.prepare()
print("Test: %d" % len(test_set.image_ids))


# Chargement du modèle en mode entraînement
model = modellib.MaskRCNN(mode = "training", config = config, model_dir = './')

# Chargement des poids du modèle
model.load_weights(
    'mask_rcnn_coco.h5', 
    by_name = True, 
    exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"]
)


# Entraînement du modèle avec un dataset d'entrée et des paramètres spécifiques prédéfinis
start_train = time.time()
model.train(train_set,test_set, learning_rate = config.LEARNING_RATE, epochs = 50, layers = "heads")
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')

history = model.keras_model.history.history
model_save = "mask_rcnn_sacv_tlb_3c_v2_res101_50epoch_lr0001_50sbe.h5"
model.keras_model.save_weights(model_save)