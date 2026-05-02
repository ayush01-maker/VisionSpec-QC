import tensorflow as tf
import numpy as np

def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(
        img_path,
        target_size=(224,224)
    )

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    return img_array
