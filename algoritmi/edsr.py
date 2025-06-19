import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)
    image = image / 255.0
    return image

def save_image(image, save_path):
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(save_path)

def edsr_super_resolution(image_path, save_path):
    model = load_model('path_to_edsr_model.h5')
    image = load_image(image_path)
    image = np.expand_dims(image, axis=0)
    upscaled_image = model.predict(image)
    save_image(upscaled_image[0], save_path)

edsr_super_resolution('inputedsr_image.jpg', 'edsr_upscaled.jpg')


