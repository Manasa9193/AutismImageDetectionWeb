import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import Xception  # Import Xception model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import os
import tensorflow as tf
img=image.load_img("C:/Users/srima/OneDrive/Documents/Autism/train/Autistic/Autistic.0.jpg")

#plt.show()
#print(cv2.imread("C:/Users/srima/OneDrive/Documents/Autism/archive (2)/AutismDataset/train/Autistic/Autistic.0.jpg"))
#print(cv2.imread("C:/Users/srima/OneDrive/Documents/Autism/train/Autistic/Autistic.0.jpg").shape)
train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)
tr_dt=train.flow_from_directory('C:/Users/srima/OneDrive/Documents/Autism/train/',
                                target_size=(200,200),batch_size=12,class_mode='binary')
val_dt=train.flow_from_directory('C:/Users/srima/OneDrive/Documents/Autism/test/',
                                target_size=(200,200),batch_size=12,class_mode='binary')
# ... (previous code for image loading and data generators)

# Load Xception model with pre-trained weights (include_top=False removes the fully connected layers)
base_model = Xception(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

# Freeze the pre-trained layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of Xception
model = tf.keras.models.Sequential([
    base_model,  # Add the Xception base model
    GlobalAveragePooling2D(),  # Convert 4D tensor output to 2D tensor
    Dense(512, activation='relu'),  # Add custom fully connected layers
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

model_fit = model.fit(tr_dt, steps_per_epoch=11, epochs=23, validation_data=val_dt)
model.save('autism.xception.model')
dir_path='C:/Users/srima/OneDrive/Documents/Autism/test/Non_Autistic.1.jpg'

imgg=image.load_img(dir_path,target_size=(200,200))
#imgg = cv2.resize(imgg, (200, 200))
x=image.img_to_array(imgg)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])
val=model.predict(images)
print(val)
threshold = 0.75  
if val >= threshold:
    print("Autistic")
else:
    print("Not Autistic")


                                   

