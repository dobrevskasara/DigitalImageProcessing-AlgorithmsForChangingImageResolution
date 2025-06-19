import cv2
def bicubic_interpolation(image_path, scale_factor):
    image = cv2.imread(image_path)
    new_dimensions = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    upscaled_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('bicubic_upscaled.jpg', upscaled_image)

bicubic_interpolation('input3_image.jpg', 0.5)
