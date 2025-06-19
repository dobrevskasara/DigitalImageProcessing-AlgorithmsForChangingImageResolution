import cv2

def bilinear_interpolation(image_path, scale_factor):
    image = cv2.imread(image_path)
    new_dimensions = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    upscaled_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('bilinear_upscaled.jpg', upscaled_image)

bilinear_interpolation('input2_image.jpg', 2.0)
