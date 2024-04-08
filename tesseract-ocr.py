import os
import json
import cv2
import pytesseract
from pytesseract import Output

def run_ocr_on_image(image_path, lang='lav'):
    """
    Runs OCR on an image and returns the extracted text.
    :param image_path: Path to the image file.
    :param lang: Language code for Tesseract OCR (default is English).
    :return: Extracted text or None if an error occurs.
    """
    try:
        blacklist_chars = '0123456789.,/\/()[]}{#$%^&*!@~`-_=+<>?;:|'
        blacklist_letters_lv = 'QWXYqwxy'
        # blacklist specific for LV and DE, ru and pl should be added
        blacklist_lv = blacklist_chars + blacklist_letters_lv
        config = f'--psm 3 --oem 3 -c tessedit_char_blacklist={blacklist_lv}'
        text = pytesseract.image_to_string(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), lang=lang, config=config)
        return text.strip()
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
        return None
    
def get_true_text(json_item, image_path):
    """
    Extracts the true text from a JSON item based on the dataset type inferred from the image path.
    
    :param json_item: The JSON item containing the details of the image.
    :param image_path: The path of the image being processed, used to determine the dataset type.
    :return: The true text extracted from the JSON item.
    """
    true_text = ""
    if 'berlin-mitte/' in image_path:
        true_text = json_item.get('description', '')
    elif 'timenote/' in image_path:
        # Concatenate the fields to form the true text for the 'timenote' dataset
        true_text = ' '.join(json_item.get(field, '') for field in ['person_name', 'extra_names', 'patronymic']).strip()
    return true_text

def get_json_details(image_path, directory):
    """
    Searches for and extracts details from a JSON file corresponding to the given image file.
    :param image_path: Path to the image file.
    :param directory: Directory to search for the JSON file.
    :return: Details from the JSON file or None if not found.
    """
    # Determine dataset type based on the presence of specific subdirectories in the image_path
    if 'timenote/' in image_path:
        # For 'timenote/' dataset, replace '_' with '/' and adjust extension to match JSON "main_image_url"
        image_file_name = os.path.basename(image_path)
        parts = image_file_name.split('_')
        if len(parts) > 2:
            # Only proceed if there are at least two underscores to replace
            formatted_image_name = '/'.join(parts[:2]) + '_' + '_'.join(parts[2:])
        else:
            # If less than two underscores, no replacement is needed
            formatted_image_name = image_file_name
        # Adjust extension to match JSON "main_image_url"
        formatted_image_name = formatted_image_name.rsplit('.', 1)[0] + '.jpg'
        print("AAA debug:", formatted_image_name)
        image_url_key = "main_image_url"
    elif 'berlin-mitte/' in image_path:
        # For 'berlin-mitte/' dataset, use the image filename as-is to match JSON "imageURL"
        formatted_image_name = os.path.basename(image_path)
        image_url_key = "imageURL"
    else:
        # If the dataset does not match known structures, return None
        return None

    for file in os.listdir(directory):
        if file.endswith(".json"):
            json_path = os.path.join(directory, file)
            with open(json_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    # Extract filename from the URL in JSON data based on the dataset type
                    json_image_file_name = os.path.basename(item.get(image_url_key, ""))
                    if formatted_image_name.endswith(json_image_file_name):
                        return item
    return None

def extract_lang(item):
    if 'nationality' in item:
        map = {
            "latvian": "lav",
            "russian": "rus",
            "pole": "pol",
            "german": "deu",
        }
        return map.get(item['nationality'].lower(), "lav")
    else:
        return "lav"
    
import string

def basic_similarity_score(ocr_text, true_text):
    """
    Compares OCR text with true text in a naive way by checking the proportion
    of matching words, disregarding the order and context.
    
    :param ocr_text: The text extracted from OCR.
    :param true_text: The actual text from the dataset.
    :return: A simple match score as a float between 0 and 1.
    """
    # Normalize texts: lowercase and remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    norm_ocr_text = ocr_text.lower().translate(translator)
    norm_true_text = true_text.lower().translate(translator)
    
    # Tokenize texts into sets of words to remove duplicates
    ocr_words = set(norm_ocr_text.split())
    true_words = set(norm_true_text.split())
    
    # Avoid division by zero if true_text is empty
    if not true_words:
        return "0.00000"
    
    # Calculate match score
    match_count = len(ocr_words.intersection(true_words))
    total_true_words = len(true_words)
    
    # Handle case where OCR text is empty
    if total_true_words == 0 or len(ocr_text.strip()) == 0:
        return "0.00000"

    match_score = match_count / total_true_words
    formatted_score = "{:.5f}".format(match_score)
    return formatted_score


def process_directory(directory):
    """
    Walks through the given directory, performing OCR on each image file,
    and then extracts additional details from a corresponding JSON file.
    :param directory: Directory to process.
    :param default_lang: Default OCR language.
    """
    lang='lav' # assume default language is Latvian
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                true_text = ""  # Default value if no JSON details are found or if an error occurs
                print(image_path)
                json_details = get_json_details(image_path, root)
                if json_details:
                    if 'timenote' in image_path:
                        lang = extract_lang(json_details)
                    else: 
                        lang = 'deu'
                    print(f"Found JSON details for {file}: {json_details}")
                    # Extract true text using the image path to determine dataset type
                    true_text = get_true_text(json_details, image_path)
                    print(f"True text for {file}: {true_text}")
                else:
                    print(f"No JSON details found for {file}")
                ocr_text = run_ocr_on_image(image_path, lang)
                if ocr_text:
                    print(f"OCR text for {file}: {ocr_text}...")

                match_score = basic_similarity_score(ocr_text, true_text)
                print(f"Basic similarity score for {file}: {match_score}")

if __name__ == "__main__":
    dataset_directory = "dataset/timenote/test"
    #dataset_directory = "dataset/berlin-mitte/" 
    process_directory(dataset_directory)
