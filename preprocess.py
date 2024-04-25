from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters
from object_selection import ObjectSelection
from tesseract_ocr import TesseractOCR
from google_vision_ocr import GoogleVisionOCR
from scipy.ndimage import interpolation as inter
import os

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

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
    corrected = image

    # sometimes can worsen google api recognition
    # best_angle, corrected = correct_skew(image)
    # print(f"Deskew angle: {best_angle}")

    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    # inversion almost always improves the quality of the OCR
    if invert:
        gray = cv2.bitwise_not(gray)

    # any denoising, thresholding, sharpening or blurring didn gave better results

    # Get the dimensions of the image
    height, width = corrected.shape[:2]

    # Calculate the resize factor
    max_dim = max(height, width)
    resize_factor = 2000 / max_dim

    # big impact on the quality of the OCR
    resized = cv2.resize(gray, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)

    # slight improvement
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(resized, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded

# Load the image from the file system
# image_path = 'dataset/timenote/Jaunciema_kapi/2018_10_Valija-Ernestsone.jpg'
# image = Image.open(image_path)

# object selection usage performs well only on timenote data subset
# object_selector = ObjectSelection(image_path, verbose=True)
# color_segmentation, edge_detection = object_selector.run()
# color_segmentation = Image.open(color_segmentation)
# edge_detection = Image.open(edge_detection)
# processed = preprocess_for_ocr(image, invert=True)
# processed_color_segmentation = preprocess_for_ocr(color_segmentation, invert=True)
# processed_edge_detection = preprocess_for_ocr(edge_detection, invert=True)

SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
# Define the directory to walk
root_dir = 'dataset/timenote/test/'
output_dir = 'dataset_preprocessed/timenote/test/'

# Walk through the directory
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        # Check if the file is an image
        if filename.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
            image_path = os.path.join(dirpath, filename)
            base_filename = os.path.splitext(filename)[0]
            base_output_dir = dirpath.replace(root_dir, output_dir)
            os.makedirs(base_output_dir, exist_ok=True)

            processed_paths = [
                os.path.join(base_output_dir, base_filename + '_processed.png'),
                os.path.join(base_output_dir, base_filename + '_processed_color_segmentation.png'),
                os.path.join(base_output_dir, base_filename + '_processed_edge_detection.png')
            ]

            # Skip if all processed files exist
            if all(os.path.exists(path) for path in processed_paths):
                print(f"Skipping {image_path}")
                continue

            image = cv2.imread(image_path)
            processed = preprocess_for_ocr(image, invert=True)
            object_selector = ObjectSelection(image_path, verbose=True)
            color_segmentation_path, edge_detection_path = object_selector.run()
            color_segmentation = Image.open(color_segmentation_path)
            edge_detection = Image.open(edge_detection_path)
            processed_color_segmentation = preprocess_for_ocr(color_segmentation, invert=True)
            processed_edge_detection = preprocess_for_ocr(edge_detection, invert=True)

            # Save the preprocessed images
            cv2.imwrite(processed_paths[0], processed)
            cv2.imwrite(processed_paths[1], processed_color_segmentation)
            cv2.imwrite(processed_paths[2], processed_edge_detection)

# tesseract = TesseractOCR()
# google_vision = GoogleVisionOCR()

# text = tesseract.run_ocr(processed, 'lav')
# gtext = google_vision.perform_ocr(processed)
# text1 = tesseract.run_ocr(processed_masked1, 'lav')
# gtext1 = google_vision.perform_ocr(processed_masked1)
# text2 = tesseract.run_ocr(processed_masked2, 'lav')
# gtext2 = google_vision.perform_ocr(processed_masked2)
# print("text", text)
# print("text1", text1)
# print("text2", text2)
# print("gtext", gtext)
# print("gtext1", gtext1)
# print("gtext2", gtext2)

# Using matplotlib to display both original and processed images side by side
# plt.figure(figsize=(20, 10))
# plt.subplot(2, 2, 1)
# plt.imshow(image)
# plt.title('Original Image')
# plt.axis('on')

# plt.subplot(2, 2, 2)
# plt.imshow(processed, cmap='gray')
# plt.title('Processed Image')
# plt.axis('on')

# plt.subplot(2, 2, 3)
# plt.imshow(masked1, cmap='gray')
# plt.title('Mask1 Image')
# plt.axis('on')

# plt.subplot(2, 2, 4)
# plt.imshow(masked2, cmap='gray')
# plt.title('Mask2 Image')
# plt.axis('on')

# plt.show()
