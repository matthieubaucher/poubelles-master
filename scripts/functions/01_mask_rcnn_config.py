# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:37:36 2021

@author: 33675
"""

class maskRcnnConfig(Config):
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
    STEPS_PER_EPOCH = 20
    
    # Learning rate
    LEARNING_RATE = 0.005
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.95
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES = 30
  