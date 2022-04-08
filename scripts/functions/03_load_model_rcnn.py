# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:47:05 2021

@author: 33675
"""

def loadModelRcnn(name):
    model_test = modellib.MaskRCNN(mode = "inference", config = config, model_dir = './')
    model_test.load_weights(name, by_name = True)
    return(model_test)