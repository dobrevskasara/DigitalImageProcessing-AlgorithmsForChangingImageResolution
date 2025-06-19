import cv2

def nearest_neighbor_interpolation(image_path, scale_factor):
    image = cv2.imread(image_path)
    new_dimensions = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    upscaled_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('nearest_neighbor_upscaled.jpg', upscaled_image)

nearest_neighbor_interpolation('input_image.jpg', 2.0)
