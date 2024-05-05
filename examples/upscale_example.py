import cv2
import matplotlib.pyplot as plt

# Load an image
image_path = 'dataset/preprocessing_test/2012_08_janis-aigars-piemineklis.jpg'
image = cv2.imread(image_path)

# Define the scale factor
scale_factor = 20  # Extreme upscaling

# Upscale the image using Lanczos interpolation
upscaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


# Convert color from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
upscaled_image_rgb = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB)

# Display the original and upscaled images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('on')

plt.subplot(1, 2, 2)
plt.imshow(upscaled_image_rgb)
plt.title(f'Upscaled Image x{scale_factor} (Bicubic Interpolation)')
plt.axis('on')
plt.show()
