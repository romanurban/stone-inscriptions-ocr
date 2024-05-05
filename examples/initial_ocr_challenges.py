import cv2
import matplotlib.pyplot as plt

# Load images
image_paths = [
    'dataset/preprocessing_test/2018_08_Vera-Pole.jpg',
    'dataset/preprocessing_test/2022_05_Voldemars-Bahmanis.jpg',
    'dataset/preprocessing_test/2022_09_Malvine-Gutmane.jpg',
    'dataset/preprocessing_test/2022_10_Arnolds-Juris-Valters.jpg',
    'dataset/preprocessing_test/2022_12_Alfreds-Zivtins.jpg',
    'dataset/preprocessing_test/saarbruecker_st_prenzl_al_c.jpg'
]

images = [cv2.imread(path) for path in image_paths]

# Plot images on a 2x3 grid
plt.figure(figsize=(12, 8))

for i, image in enumerate(images, start=1):
    plt.subplot(2, 3, i)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'{chr(96+i)})')  # Using chr(96+i) to get 'a', 'b', 'c', ...
    plt.axis('off')

plt.tight_layout()
plt.show()