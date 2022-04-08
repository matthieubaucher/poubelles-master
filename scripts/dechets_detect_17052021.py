# Chargement des librairies
from os import chdir, getcwd, listdir
from os.path import isfile, join, splitext

#chdir('C:/Users/Alexandre.Iborra/Desktop/Véolia/sac_detect')
#chdir('C:/Users/Alexandre.Iborra/Desktop/Véolia/kangaroo')

#chemin = 'C:/Users/alexi/Desktop/Véolia/sac_detect'
chemin = 'C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/exemple_kangaroo/Mask_RCNN'
chemin_img = 'C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/exemple_kangaroo/kangaroo-master'

chdir(chemin)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
import re
import pandas as pd


from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
%matplotlib inline
from os import listdir
from xml.etree import ElementTree
import time
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_config"
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # number of classes (we would normally add +1 for the background)
     # kangaroo + BG
    NUM_CLASSES = 3 + 1
    
    #BACKBONE = "resnet50"
    BACKBONE = "resnet101"
   
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 993
    
    # Learning rate
    LEARNING_RATE = 0.005
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.80
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES = 30
    
config = myMaskRCNNConfig()
config.display()


class KangarooDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        
        # Add classes. We have only one class to add.
        self.add_class("dataset", 1, "sac")
        self.add_class('dataset', 2, "cart")
        self.add_class('dataset', 3, "bout")
        
        print(dataset_dir + '/images/')
        # define data locations for images and annotations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        
        # Iterate through all files in the folder to 
        #add class, images and annotaions
        for filename in listdir(images_dir):
            
            # extract image id
            #image_id = filename[:-4]
            temp = re.findall("\d+", filename)
            test_is_train = re.split("\_", filename)[0] # On extrait le préfix du nom de l'image pour savoir si c'est train ou validation

            image_id = temp[0]
                        
            # skip bad images
            #if image_id in ['00090']:
               # continue
            # skip all images after 150 if we are building the train set
            if is_train and test_is_train == "val":
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and test_is_train == "train":
                continue
            
            # setting image file
            img_path = images_dir + filename
            
            # getting prefix images
            pref_img = re.split("\_", filename)[1]
            
            # setting annotations file
            ann_path = annotations_dir + pref_img + '_' + image_id + '.xml'
            
            # adding images and annotations to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
            
# extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        classes = []
        lst_class = root.findall('.//name') # liste des classes présentes dans le xml

        box_id = 0
        
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
            classes.append(self.class_names.index(lst_class[box_id].text))
            box_id = box_id + 1
        
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, classes, width, height
# load the masks for an image
    """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
     """
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        
        # define anntation  file location
        path = info['annotation']
        
        # load XML
        boxes, classes, w, h = self.extract_boxes(path)
       
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            #class_ids.append(self.class_names.index('sac'))
        return masks, np.asarray(classes, dtype='int32')
# load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']
    

# Fonction pour relancer le modèle en mode inférence et charger les poids de notre modèle déjà entrainé    
def reload_model(name):
    
    model_test = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
    model_test.load_weights(name, by_name=True)
    return(model_test)
    
# Fonction qui permet d'afficher les images  d'un répertoire, avec la prédiction (zone + score)   
def test_result(path,model_detect):
    
    path_image = path
    
    for image in listdir(path_image):
        image_test = load_img(path + image)
        image_test = img_to_array(image_test)
        result = model_detect.detect([image_test])
        r = result[0]
        visualize.display_instances(image_test, r['rois'], r['masks'], r['class_ids'],
                                    ["0", "sac", "cart", "bout"], r['scores'],
                                    title="Predictions")

# Fonction pour avoir le tableau des predictions ainsi que matrice de confusion + classification_report de sklearn
def metrics_model(path,model_detect,conf):
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


####### Train - test


train_set = KangarooDataset()
train_set.load_dataset(chemin_img, is_train=True)
train_set.prepare()
print("Train: %d" % len(train_set.image_ids))

# prepare test/val set
test_set = KangarooDataset()
test_set.load_dataset(chemin_img, is_train=False)
test_set.prepare()
print("Test: %d" % len(test_set.image_ids))


####### Train modèle
print("Loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="training", config=config, model_dir='./')

#load the weights for COCO
model.load_weights('mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])


## train heads with higher lr to speedup the learning
#model.train(data_set, test_set, learning_rate=config.LEARNING_RATE, epochs=30, layers="heads")

start_train = time.time()
model.train(train_set,test_set, learning_rate = config.LEARNING_RATE, epochs = 20, layers = "heads")
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')

history = model.keras_model.history.history


model_path = 'mask_rcnn_sacv_tlb_3c_v2_res101_20epoch_lr0005_1093'   + '.h5'

model.keras_model.save_weights(model_path)



##### Test modèle


  
model_path = '../weights/mask_rcnn_sacv_tlb_3c_v2_res101_20epoch_lr0005_1093'   + '.h5'
path = model_path

model_test = loadModelRcnn(path)

matc,classr,tabl = metrics_model(chemin_img + '/test/', model_test, config.DETECTION_MIN_CONFIDENCE)
pd.crosstab(tabl["y"],tabl["y_pred"])

print(matc)
print(classr)
pd.crosstab(tabl["y"],tabl["y_pred"])

test_result(chemin_img + '/test4/', model_test)

tabl.to_excel(chemin + '/export_result_tlb.xlsx', index = False, header=True)

kc_101.jpg empty
kc_117.jpg 0,9096