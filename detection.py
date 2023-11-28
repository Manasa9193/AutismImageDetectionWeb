import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.preprocessing import image
from keras.optimizers import RMSprop
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy

import cv2
import os
img=image.load_img("C:/Users/srima/OneDrive/Documents/Autism/train/Autistic/Autistic.0.jpg")
plt.imshow(img)
#plt.show()
#print(cv2.imread("C:/Users/srima/OneDrive/Documents/Autism/archive (2)/AutismDataset/train/Autistic/Autistic.0.jpg"))
#print(cv2.imread("C:/Users/srima/OneDrive/Documents/Autism/train/Autistic/Autistic.0.jpg").shape)
train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)
tr_dt=train.flow_from_directory('C:/Users/srima/OneDrive/Documents/Autism/train/',
                                target_size=(200,200),batch_size=12,class_mode='binary')
val_dt=train.flow_from_directory('C:/Users/srima/OneDrive/Documents/Autism/test/',
                                target_size=(200,200),batch_size=12,class_mode='binary')
#print(tr_dt.class_indices)
model =tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',
                                   input_shape=(200,200,3)),tf.keras.layers.MaxPool2D(2,2),
                                   #
                                   tf.keras.layers.Conv2D(32,(3,3),activation='relu',
                                   input_shape=(200,200,3)),tf.keras.layers.MaxPool2D(2,2),
                                   #
                                   tf.keras.layers.Conv2D(64,(3,3),activation='relu',
                                   input_shape=(200,200,3)),tf.keras.layers.MaxPool2D(2,2),
                                   ##
                                   tf.keras.layers.Flatten(),
                                   ##
                                   tf.keras.layers.Dense(512,activation='relu'),
                                   ##
                                   tf.keras.layers.Dense(1,activation='sigmoid')
                                    ])
model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['accuracy'])
model_fit=model.fit(tr_dt,steps_per_epoch=12,epochs=50,validation_data=val_dt)
model.save('autism.cnn.model')
dir_path='C:/Users/srima/OneDrive/Documents/Autism/test/Non_Autistic.1.jpg'

imgg=image.load_img(dir_path,target_size=(200,200))
#imgg = cv2.resize(imgg, (200, 200))
x=image.img_to_array(imgg)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])
val=model.predict(images)
if val==0:
        print("Remember, a diagnosis doesn't define you. Your unique strengths and perspectives shine through. You're not alone—there's a community here to support and uplift you every step of the way. ")
if val==1:
          print("Not_Autistic")


                                   

