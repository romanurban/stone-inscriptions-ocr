from similarity_metrics import basic_similarity_score
from tesseract_ocr import run_tesseract_ocr
from dataset_helper import get_true_text, get_json_details, extract_lang
import os

DEFAULT_LANGUAGE = 'lav' # default latvian language code for timenote dataset
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
MITTE_DS_LANG_CODE = 'deu' # default german language code for the 'berlin-mitte/' dataset

def process_directory(directory):
    """
    Walks through the given directory, performing OCR on each image file,
    and then extracts additional details from a corresponding JSON file.
    :param directory: Directory to process.
    :param default_lang: Default OCR language.
    """
    lang = DEFAULT_LANGUAGE
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                image_path = os.path.join(root, file)
                true_text = ""  # Default value if no JSON details are found or if an error occurs
                print(image_path)
                json_details = get_json_details(image_path, root)
                if json_details:
                    if 'timenote' in image_path:
                        lang = extract_lang(json_details)
                    else: 
                        lang = MITTE_DS_LANG_CODE
                    print(f"Found JSON details for {file}: {json_details}")
                    # Extract true text using the image path to determine dataset type
                    true_text = get_true_text(json_details, image_path)
                    print(f"True text for {file}: {true_text}")
                else:
                    print(f"No JSON details found for {file}")
                ocr_text = run_tesseract_ocr(image_path, lang)
                if ocr_text:
                    print(f"OCR text for {file}: {ocr_text}...")

                match_score = basic_similarity_score(ocr_text, true_text)
                print(f"Basic similarity score for {file}: {match_score}")

if __name__ == "__main__":
    dataset_directory = "dataset/timenote/test"
    #dataset_directory = "dataset/berlin-mitte/" 
    process_directory(dataset_directory)
