from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters
from object_selection import ObjectSelection
from tesseract_ocr import TesseractOCR
from google_vision_ocr import GoogleVisionOCR

def preprocess_for_ocr(image, invert=True):
    """
    Preprocesses an image for OCR by enhancing the contrast between dark text and a light background.

    Parameters:
    - image: The input image.

    Returns:
    - The preprocessed image.
    """

    # Convert the image to grayscale
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the resize factor
    max_dim = max(height, width)
    resize_factor = 2000 / max_dim

    # big impact on the quality of the OCR
    resized = cv2.resize(gray, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)

    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(resized, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded

# Load the image from the file system
# image_path = 'dataset/preprocessing_test/2016_08_Harijs-Upitis.jpg'
image_path = 'dataset/preprocessing_test/2022_11_Skaidrite-Riekstina.jpg'
image = Image.open(image_path)

object_selector = ObjectSelection(image_path, verbose=False)
masked1_path, masked2_path = object_selector.run()
masked1 = Image.open(masked1_path)
masked2 = Image.open(masked2_path)
processed = preprocess_for_ocr(image, invert=True)
processed_masked1 = preprocess_for_ocr(masked1, invert=True)
processed_masked2 = preprocess_for_ocr(masked2, invert=True)

tesseract = TesseractOCR()
google_vision = GoogleVisionOCR()

text = tesseract.run_ocr(processed, 'lav')
gtext = google_vision.perform_ocr(processed)
text1 = tesseract.run_ocr(processed_masked1, 'lav')
gtext1 = google_vision.perform_ocr(processed_masked1)
text2 = tesseract.run_ocr(processed_masked2, 'lav')
gtext2 = google_vision.perform_ocr(processed_masked2)
print("text", text)
print("text1", text1)
print("text2", text2)
print("gtext", gtext)
print("gtext1", gtext1)
print("gtext2", gtext2)

# Using matplotlib to display both original and processed images side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(processed, cmap='gray')
plt.title('Processed Image')
plt.axis('off')

plt.show()
