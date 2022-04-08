# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 11:01:59 2021

@author: 33675
"""

# Chargement des librairies
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros, asarray
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
from os import chdir, getcwd, listdir
from os.path import isfile, join, splitext
import time
import re
import pandas as pd
import glob
import keras
from keras.models import Sequential
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img, ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from string import Template
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
#%matplotlib inline
from xml.etree import ElementTree
import io
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
import smtplib
import ssl
from email.mime.text import MIMEText
from email.utils import formataddr
from email.mime.multipart import MIMEMultipart  # New line
from email.mime.base import MIMEBase  # New line
from email import encoders  # New line
from email.message import EmailMessage
from email.utils import make_msgid
import mimetypes
from email.header import Header
from PIL import Image
import PIL
import tensorflow.compat.v1 as tf
from skimage import transform



#chemin_appli = "C:/Users/Alexandre.Iborra/Desktop/Veolia/application"
chemin_appli = "C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application"


# Spécification du répertoire de travail
chdir(chemin_appli + "/scripts/functions")

# Chargement des paramètres de l'application
exec(open("../../config/param.py", encoding="utf-8").read())

# Chargement des fonctions qui permettent de créer/exploiter le modèle
exec(open("01_mask_rcnn_config.py").read())
exec(open("02_build_train_dataset.py").read())
exec(open("03_load_model_rcnn.py").read())
exec(open("04_view_predicted_images.py").read())
exec(open("05_view_confusion_matrix.py").read())
exec(open("06_extract_video_img.py").read())
exec(open("07_predict_img.py").read())
exec(open("08_save_predicted_img.py").read())
exec(open("09_visualized_py_v2.py").read())
exec(open("10_send_mail.py").read())
exec(open("11_load_model_mobilenet.py").read())
exec(open("13_blur_face.py").read())


# Chargement de la configuration du modèle
config = maskRcnnConfig()
# config.display()

