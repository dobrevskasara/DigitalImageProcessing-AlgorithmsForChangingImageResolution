import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model


def srcnn_model():
    input_layer = Input(shape=(None, None, 1))

    x = Conv2D(64, (9, 9), activation='relu', padding='same')(input_layer)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = Conv2D(1, (5, 5), activation='linear', padding='same')(x)

    model = Model(inputs=input_layer, outputs=x)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32) / 255.0
    return image


def upscale_image(image, scale):
    height, width = image.shape
    new_height, new_width = height * scale, width * scale
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def display_images(lr_image, sr_image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Low Resolution")

    plt.imshow(lr_image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Super Resolution")
    plt.imshow(sr_image, cmap='gray')
    plt.show()


if __name__ == "__main__":
    lr_image = preprocess_image('low_res_image.jpg')
    upscaled_image = upscale_image(lr_image, 3)
    model = srcnn_model()
    sr_image = model.predict(upscaled_image[np.newaxis, :, :, np.newaxis])[0, :, :, 0]
    display_images(upscaled_image, sr_image)
