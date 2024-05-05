import cv2
import numpy as np
import matplotlib.pyplot as plt

color_image = cv2.imread('dataset/preprocessing_test/2016_06_E-Kampe.jpg')

gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

ret, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow(color_image)
ax[0].set_title('Colorful Image')
ax[0].axis('on')

ax[1].imshow(otsu_thresh, cmap='gray')
ax[1].set_title('Binarized Image')
ax[1].axis('on')
plt.show()