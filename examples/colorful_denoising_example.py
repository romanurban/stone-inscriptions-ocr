import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the color image
color_image = cv2.imread('dataset/preprocessing_test/2016_06_E-Kampe.jpg')

# Apply Gaussian Blurring
gaussian_blurred = cv2.GaussianBlur(color_image, (55, 55), 0)  # Kernel size (55,55) and sigmaX=0 (auto)

# Apply Median Blurring
median_blurred = cv2.medianBlur(color_image, 55)  # Kernel size 55

# Display the original, Gaussian blurred, and median blurred images
fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # Adjust subplot parameters for better display
ax[0].imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
ax[0].set_title('Original Image')
ax[0].axis('on')  # Hide axes for cleaner presentation

ax[1].imshow(cv2.cvtColor(gaussian_blurred, cv2.COLOR_BGR2RGB))
ax[1].set_title('Gaussian Blurred Image')
ax[1].axis('on')

ax[2].imshow(cv2.cvtColor(median_blurred, cv2.COLOR_BGR2RGB))
ax[2].set_title('Median Blurred Image')
ax[2].axis('on')

plt.show()
