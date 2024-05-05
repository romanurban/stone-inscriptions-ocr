import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the color image
color_image = cv2.imread('dataset/preprocessing_test/2016_06_E-Kampe.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding to binarize the image
ret, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Define a structuring element for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Perform morphological closing
closed_image = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)

# Display the original, binarized, and closed images
fig, ax = plt.subplots(1, 2, figsize=(12, 4))  # Adjust subplot parameters for better display
# ax[0].imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
# ax[0].set_title('Colorful Image')
# ax[0].axis('off')  # Hide axes for cleaner presentation

ax[0].imshow(otsu_thresh, cmap='gray')
ax[0].set_title('Binarized Image')
ax[0].axis('on')

ax[1].imshow(closed_image, cmap='gray')
ax[1].set_title('Denoised Image')
ax[1].axis('on')

plt.show()
