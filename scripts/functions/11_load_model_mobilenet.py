# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:11:23 2021

@author: 33675
"""

# Architecture du modèle mobile net pour faire de la classification d'images
def loadModelMobileNet():
    mobile_v2 = MobileNetV2(include_top = False, input_shape = (224, 224, 3))

    layers_to_freeze = 80

    for layer in mobile_v2.layers[:layers_to_freeze]:
        layer.trainable = False

    model = Sequential([mobile_v2, GlobalAveragePooling2D(), Dense(1, activation = 'sigmoid')])

    # As we are training the top layers of the MobileNet, lr stays low
    model.compile(
        loss = BinaryCrossentropy(from_logits = True), optimizer = Adam(1e-5),
        metrics = ['accuracy']
    )
    
    model.load_weights('transfer_mobile_freeze_80_wVeolia_noflat2.h5')
        
    return(model)