import cv2
import numpy as np

def stack_images(image_paths):
    images = [cv2.imread(path) for path in image_paths]
    stacked_image = np.mean(images, axis=0).astype(np.uint8)
    cv2.imwrite('stacked_image.jpg', stacked_image)

stack_images(['image.jpg', 'image2.jpg', 'image3.jpg'])
