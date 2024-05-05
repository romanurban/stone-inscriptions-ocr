import os
import json

class AppleVisionOCR:
    def __init__(self, base_directory):
        """Initializes the AppleVisionOCR class with a base directory for OCR results.

        Args:
            base_directory (str): The base directory where OCR results are stored.
        """
        self.base_directory = base_directory

    def perform_ocr(self, image_url):
        """Performs OCR by finding a JSON file corresponding to the image URL.

        Args:
            image_url (str): The URL of the image, used to determine the path to the OCR result.

        Returns:
            str: The OCR text detected in the image.
        """
        # Extract the path from the URL
        path_parts = image_url.split('/')
        # Remove the filename from the path to reach the directory
        if 'dataset' in path_parts:
            directory_path = '/'.join(path_parts[path_parts.index('dataset'):-1])
        elif 'dataset_preprocessed' in path_parts:
            directory_path = '/'.join(path_parts[path_parts.index('dataset_preprocessed'):-1])
        else:
            return "Invalid image URL."

        # Construct the path to the JSON file in the corresponding directory
        json_file_path = os.path.join(self.base_directory, directory_path, 'ocr_result.json')

        # Read the JSON file and find the corresponding OCR result
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as file:
                data = json.load(file)
                # Iterate over each entry in the JSON data
                for entry in data:
                    # Check if the filename in the JSON matches the image file in the URL
                    if image_url.endswith(entry['filename']):
                        return entry['detected']
        return ""

# Example of how to use the AppleVisionOCR class
# base_dir = 'ocr_results/apple_vision_source_init'
# apple_vision_ocr = AppleVisionOCR(base_dir)
# result = apple_vision_ocr.perform_ocr('/dataset/berlin-mitte/AErzte_ohne_Grenzen.jpg')
# print(result)