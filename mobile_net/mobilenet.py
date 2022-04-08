from keras.models import Sequential
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import io


mobile_v2 = MobileNetV2(include_top=False, input_shape=(224,224,3))

#choose how many layers to freeze, 80~120 same res on this task
layers_to_freeze = 80

for layer in mobile_v2.layers[:layers_to_freeze]:
    layer.trainable = False

model = Sequential([mobile_v2, GlobalAveragePooling2D(), Dense(1, activation='sigmoid')])

model.summary()

#As we are training the top layers of the MobileNet, lr stays low
model.compile(
    loss= BinaryCrossentropy(from_logits=True), optimizer=Adam(1e-5),
    metrics=['accuracy'])

#batch size capacity depends on your GPU
batch_size = 16

# Now using data augmentation, delete 3 last attributes if you want
# to do it with no augmentation 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    r'D:\Talan\clean_dirty_classifier\data\train2',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = train_datagen.flow_from_directory(
   r'D:\Talan\clean_dirty_classifier\data\test2',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator2 = train_datagen.flow_from_directory(
    r'D:\Talan\clean_dirty_classifier\data\val2',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)
#uncomment if you want to re-train a model
# transfer_mod = model.fit_generator(
#     train_generator,
#     steps_per_epoch=30,
#     epochs=40,
#     validation_data=test_generator,
#     validation_steps=10)


# load weights if already existing, if error maybe providing of
# maybe from 13-16 where trainable layers are chosen
# model.load('transfer_mobile_freeze_80_2')
# transfer_mod = model.load_weights('transfer_mobile_freeze_80_wVeolia.h5')



# if training on a new set-up you can save weights:
model.load_weights('transfer_mobile_freeze_80_wVeolia_noflat2.h5')




############################### Print Results ############################### 
#plot learning curve
# import matplotlib.pyplot as plt

# plt.figure()
# plt.title('Transfered MobileNetV2 (steps per ep:30, val. steps:10, frozen layers:60)')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.plot(transfer_mod.history.get('accuracy'))
# plt.show()


y_pred = model.predict(test_generator2) 
y_pred = (y_pred > 0.1)*1
y_pred = y_pred.ravel()
y_pred_list = []
y_true_list = []

for i in range(y_pred.shape[0]):
    if y_pred[i] == 0:
        y_pred_list.append('clean')
    else: y_pred_list.append('dirty')

y_true = test_generator2.classes

for i in range(y_true.shape[0]):
    if y_true[i] == 0:
        y_true_list.append('clean')
    else: y_true_list.append('dirty')



# #### print reports
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

cmtx = pd.DataFrame(
    confusion_matrix(y_true_list, y_pred_list, labels=['clean', 'dirty']), 
    index=['true:clean', 'true:dirty'], 
    columns=['pred:clean', 'pred:dirty']
)
print(cmtx)


print(classification_report(y_true_list, y_pred_list, target_names=['clean', 'dirty']))