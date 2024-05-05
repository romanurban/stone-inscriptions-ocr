import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Create a colorful image using PIL
image = Image.new('RGB', (10, 10), 'white')  # Create a white background image
draw = ImageDraw.Draw(image)
draw.rectangle([1, 1, 3, 3], fill='red')
draw.rectangle([6, 1, 8, 3], fill='green')
draw.rectangle([1, 6, 3, 8], fill='blue')
draw.rectangle([6, 6, 8, 8], fill='yellow')

# Convert the image to grayscale
gray_image = image.convert('L')

# Binarize the grayscale image using a threshold and map to 0-255 scale
threshold = 128  # Define the threshold
binarized_image = gray_image.point(lambda x: 255 if x > threshold else 0, 'L')  # Use 'L' mode for 8-bit pixels

# Display the colorful, grayscale, and binarized images
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow(image)
ax[0].set_title('Colorful Image')
ax[0].axis('on')

ax[1].imshow(gray_image, cmap='gray')
ax[1].set_title('Grayscale Image')
ax[1].axis('on')

plt.show()

# Convert gray image to numpy array and print the pixel values
gray_array = np.array(gray_image)
print("gray values:")
print(gray_array)
