import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model
def load_and_predict(image_path):
    model_path = 'saved_models/autism_cnn_model.h5'
    model = load_model(model_path)




def load_and_predict(image_path):
    model_path = "C:/Users/srima/OneDrive/Documents/Autism/autism.cnn.model"
    model = load_model(model_path)

    img = image.load_img(image_path, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    prediction = model.predict(images)
    return prediction
