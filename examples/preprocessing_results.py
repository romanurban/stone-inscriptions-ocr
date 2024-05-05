import cv2
import matplotlib.pyplot as plt


# Load images
image_paths = [
    'dataset/preprocessing_test/results/2022_06_Peteris-Terauds.jpg',
    'dataset/preprocessing_test/results/2022_06_Peteris-Terauds_processed_edge_detection.png',
    'dataset/preprocessing_test/results/2022_10_Arnolds-Juris-Valters.jpg',
    'dataset/preprocessing_test/results/2022_10_Arnolds-Juris-Valters_processed_color_segmentation.png',
    'dataset/preprocessing_test/results/2022_12_Alfreds-Zivtins.jpg',
    'dataset/preprocessing_test/results/2022_12_Alfreds-Zivtins_processed_color_segmentation.png'
]

images = [cv2.imread(path) for path in image_paths]

# Plot images on a 3x2 grid
plt.figure(figsize=(12, 8))

for i, image in enumerate(images, start=1):
    plt.subplot(3, 2, i)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if i % 2 == 1:
        plt.title('Pirms')
    else:
        plt.title('Pēc')
    plt.axis('off')

plt.tight_layout()
plt.show()
