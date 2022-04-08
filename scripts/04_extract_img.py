# Importation des librairies
import cv2
import os
from mrcnn.config import Config
from mrcnn import visualize
from mrcnn import model as modellib
from keras.preprocessing.image import img_to_array
import smtplib
import ssl


# Modification du repertoire de travail
#path = "C:/Users/alexi/Desktop/Véolia/Extract_img"
#path_img = "C:/Users/alexi/Desktop/Véolia/Extract_img/Image"
path = chemin_appli + "/data"
os.chdir(path)
### Fonctions pour la récupartion automatique d'image

# Paramètre pour le modèle
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
    STEPS_PER_EPOCH = 574
    
    # Learning rate
    LEARNING_RATE = 0.001
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.60
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES = 30

# Lecture du modèle
def load_model(name):
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
    model.load_weights(name, by_name=True)
    return(model)

# Test + Affichage des predictions
def Extract_predict(sec):
    vid.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vid.read()
    cpt = 0
    
    # Paramètre mail
    #smtp_adress = 'smtp.gmail.com'
   # smtp_port = 465
    # on rentre les informations sur notre adresse e-mail
    #email_adress = 'alexiborra7@gmail.com'
    #email_password = ''

    # on rentre les informations sur le destinataire
   # email_receiver = 'alex.iborra@orange.fr'
    
    if hasFrames:
        img = img_to_array(image)
        result = model.detect([img])
        r = result[0]
        sc = r["scores"]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],                                    
                            ["0","sac", "cart", "bout"], r['scores'],
                            title="Predictions")
        
        #if sc.size >= 1 and cpt < 2:
            #print("detect mais pas alert")
            # on crée la connexion
            #context = ssl.create_default_context()
            #with smtplib.SMTP_SSL(smtp_adress, smtp_port, context=context) as server:
              # connexion au compte
              #server.login(email_adress, email_password)
              # envoi du mail
              #server.sendmail(email_adress, email_receiver, 'test mail')
        #elif sc.size >= 1 and cpt < 2:
    return hasFrames

# Renvoie une image s'il elle existe pour une seconde donnée
def extract_img(sec):
    vid.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vid.read()
    return (hasFrames,image)

# Renvoie la prediction pour une image de la vidéo
def extract_predict(image,cpt):
    img = img_to_array(image)
    result = model.detect([img])
    r = result[0]
    sc = r["scores"]
    if sc.size >= 1 :
        cpt = cpt + 1
    else :
        cpt = 0
    return(sc.size,cpt)
    
# Envoie un email (ajout email dans param)
def send_email():
    # Paramètre mail
    smtp_adress = 'smtp.gmail.com'
    smtp_port = 465
    # on rentre les informations sur notre adresse e-mail
    email_adress = 'talansolutions@gmail.com'
    email_password = 'talansolutions2021'

    # on rentre les informations sur le destinataire
    email_receiver = 'tanguy.leblevenec@gmail.com'
    # on crée la connexion
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_adress, smtp_port, context=context) as server:
              # connexion au compte
              server.login(email_adress, email_password)
              # envoi du mail
              server.sendmail(email_adress, email_receiver, 'test mail')
    

# Récupération du modèle pour prédiction
config = myMaskRCNNConfig()
model_path = 'C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/exemple_kangaroo/Mask_RCNN/mask_rcnn_sacv_tlb_3c_res101_20epoch_lr00005_gt30' + '.h5'
model = load_model(model_path)

# Lecture de la vidéo
vid = cv2.VideoCapture(path + "/video1.MP4")
#vid = cv2.VideoCapture(path + "/Video/test.MP4")


# Envoie mail
sec = 0
intervalle = 30 # image toutes les 30 secondes
cpt = 0 # Compteur detection
test, image = extract_img(sec) # On récupère l'image si elle existe
while test :
    pred, cpt = extract_predict(image, cpt) # On prédit sur l'image
    if cpt == 3 : # Si c'est la deuxième fois d'affilé qu'on observe un déchet alors envoie d'un mail
        print("envoie mail")
        send_email()
        cpt = 0 # Une fois le mail envoyé on remet le compteur à 0
    sec = sec + intervalle #
    sec = round(sec, 2)
    test, image = extract_img(sec)
    

# Affichage des images avec predictions
sec = 0
frameRate = 30 #//it will capture image in each 0.5 second
count= 1
success = Extract_predict(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = Extract_predict(sec)


























## Test

vid.set(cv2.CAP_PROP_POS_MSEC,30*1000)
hasFrames,image = vid.read()

test, image = extract_img(190)

image_test = img_to_array(image)
result = model.detect([image_test])
r = result[0]

test = visualize.display_instances(image_test, r['rois'], r['masks'], r['class_ids'],                                    
                            ["0","sac", "cart", "bout"], r['scores'],
                            title="Predictions")


image = image_test
boxes = r['rois'] 
masks = r['masks']
class_ids = r['class_ids']
class_names = ["0","sac", "cart", "bout"]
scores = r['scores']


n = boxes.shape[0]

#test = image_test, r['rois'], r['masks'], r['class_ids'],["0","sac", "cart", "bout"], r['scores'],title="Predictions"
visualize.display_instances(test)

visualize.display_instances(image_test, r['rois'], r['masks'], r['class_ids'],                                    
                            ["0","sac", "cart", "bout"], r['scores'],
                            title="Predictions")

sec = 0
frameRate = 30 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec)
while success: 
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

# Récupération des Frames
def getFrame(sec):
    vid.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vid.read()
    if hasFrames:
        cv2.imwrite("image"+str(count)+".jpg", image) 
        #cv2.imwrite(os.path.join(path_img , "image"+str(count)+".jpg"), image) # save frame as JPG file
    return hasFrames


# Source
#https://medium.com/@iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a128a481
#https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
