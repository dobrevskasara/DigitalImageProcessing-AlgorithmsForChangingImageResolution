import numpy as np
import cv2
import matplotlib.pyplot as plt


def lanczos_resample(image, scale_factor, a=3):
    def sinc(x):
        return np.sinc(x / np.pi)

    def lanczos_weight(x, a):
        if x == 0:
            return 1.0
        elif -a <= x <= a:
            return sinc(x) * sinc(x / a)
        else:
            return 0.0

    height, width = image.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    resampled_image = np.zeros((new_height, new_width), dtype=np.float32)

    for i in range(new_height):
        for j in range(new_width):
            x = i / scale_factor
            y = j / scale_factor
            x_int = int(np.floor(x))
            y_int = int(np.floor(y))
            total_weight = 0.0
            interpolated_value = 0.0

            for m in range(x_int - a + 1, x_int + a):
                for n in range(y_int - a + 1, y_int + a):
                    if 0 <= m < height and 0 <= n < width:
                        weight = lanczos_weight(x - m, a) * lanczos_weight(y - n, a)
                        interpolated_value += image[m, n] * weight
                        total_weight += weight

            resampled_image[i, j] = interpolated_value / total_weight if total_weight != 0.0 else image[x_int, y_int]

    return resampled_image.astype(np.uint8)


original_image = cv2.imread('path_to_your_image', cv2.IMREAD_GRAYSCALE)
if original_image is None:
    original_image = np.random.rand(10, 10) * 255

plt.figure(figsize=(6, 6))
plt.imshow(original_image, cmap='gray')
plt.title('Оригинална слика')
plt.axis('off')
plt.show()

scale_factor = 2
resampled_image = lanczos_resample(original_image, scale_factor)

plt.figure(figsize=(6, 6))
plt.imshow(resampled_image, cmap='gray')
plt.title(f'Ресемплирана слика (фактор на зголемување: {scale_factor})')
plt.axis('off')
plt.show()
