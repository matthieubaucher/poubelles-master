# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:01:45 2021

@author: 33675
"""
# Rcnn / objet detection

# Définir le répertoire qui contient les images et annotations
chemin_img = chemin_appli + "/data/evaluation"
chdir(chemin_appli + "/model")
model_load = "../weights/mask_rcnn_sacv_tlb_3c_v2_res101_20epoch_lr0005_877sbe.h5"


# Charger le modèle
model_test = loadModelRcnn(model_load)

# Affichage des images avec la zone de prédiction
#viewPredImg(chemin_img + '/test4/', model_test)

# Matrice de confusion et metrics pour un modèle et chemin d'image donné
matc, classr,tabl = confusionMatrix(chemin_img + '/test4/', model_test, config.DETECTION_MIN_CONFIDENCE)
pd.crosstab(tabl["y"], tabl["y_pred"])


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
        
# Chargement du modèle de détection d'objets
model_test = loadModelRcnn("weights/mask_rcnn_sacv_tlb_3c_v2_res101_20epoch_lr0005_877sbe.h5")

matc,classr,tabl = metrics_model(chemin_img + '/test/', model_test, config.DETECTION_MIN_CONFIDENCE)
pd.crosstab(tabl["y"],tabl["y_pred"])

print(matc)
print(classr)
pd.crosstab(tabl["y"],tabl["y_pred"])

test_result('data/demonstration/dechet_detect/', model_test)



# Mobile net / classification
model = loadModelMobileNet()

image_clean1 = load_img("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test/c1.PNG")
image_clean2 = load_img("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test/c2.PNG")
image_clean3 = load_img("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test/c3.PNG")
image_clean4 = load_img("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test/c4.PNG")
image_clean5 = load_img("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test/c5.PNG")

image_dirty1 = load_img("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test/d1.JPEG")
image_dirty2 = load_img("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test/d2.JPEG")
image_dirty3 = load_img("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test/d3.JPEG")
image_dirty4 = load_img("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test/d4.JPG")
image_dirty5 = load_img("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test/d5.JPG")

image = image_clean1


image = image_dirty4


 image = load('C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test/video_image_dechet.PNG')
 image = load('C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test2/dirty/dirty5.PNG')

 model.predict(image)

imgv1 = img_to_array(image).astype('float32')
imgv2 = cv2.resize(imgv1, (224,224))
imgv3 = np.expand_dims(imgv2, axis=0)
y_pred = model.predict(imgv3).astype('float')
y_pred

y_pred = (y_pred < 0.1) * 1
y_pred


y_pred = y_pred.ravel()


y_pred = y_pred.ravel()
y_pred_list = []
y_true_list = []

y_pred == 1

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

 image = load('C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test2/dirty/dirty4.png')

"C::Users\33675\Documents\Professionel\Talan\Projets\Veolia\application\data\demonstration\image_test3\dirty"

np_image = np.array(image).astype('float32') / 255
np_image = transform.resize(np_image, (224, 224, 3))
np_image = np.expand_dims(np_image, axis = 0)

model.predict(image)
y_pred

model.predict(test_generator2).astype(float).round(3)


sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

batch_size = 32
img_height = 224
img_width = 224

img = keras.preprocessing.image.load_img(
    'C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test3/dirty/d5.jpg',
    target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
model.predict(img_array, steps=1)


score = tf.nn.softmax(predictions[0])


test_generator2 = test_datagen.flow_from_directory(
    r'C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application/data/demonstration/image_test4/',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)


model.predict_generator(test_generator2)


C:\Users\33675\Documents\Professionel\Talan\Projets\Veolia\application\data\demonstration\image_test4

directory=pred_dir,
    target_size=(28, 28),
    color_mode="rgb",
    batch_size=32,
    class_mode=None,
    shuffle=False