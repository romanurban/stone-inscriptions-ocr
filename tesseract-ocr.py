import os
import json
from PIL import Image
import cv2
import pytesseract
from pytesseract import Output

### note for the future
# i can make an estimate of needed rotation using tesseract and perform that

# baseline OCR function
def run_ocr_on_image(image_path, lang):
    try:
        # whitespaces are ignored due to a bug https://github.com/tesseract-ocr/tesseract/issues/2923
        whitelist_chars_lv = 'AĀBCČDEĒFGĢHIĪJKĶLĻMNŅOPRSŠTUŪVZŽaābcčdeēfgģhiījkķlļmnņoprsštuūvzž'
        blacklist_chars = '0123456789.,/\/()[]}{#$%^&*!@~`-_=+<>?;:|' # blacklist doesn't have such problem
        blacklist_letters_lv = 'QWXYqwxy'
        blacklist_lv = blacklist_chars + blacklist_letters_lv
        config_w_blacklist = f'--psm 3 --oem 3 -c tessedit_char_blacklist={blacklist_lv}'
        # there is a difference at least PIL uses RGB and cv2 BRG color mode
        # another difference is cli tesseract, it uses leptonica library for image processing, probably that is the difference 
        # openCV also uses Leptonica so that is most close to cli tesseract lib
        # maybe to get rid of deffects caused by image library - we need to execute tesseract from cli using python
        # and not use pytesseract wrapper
        #text = pytesseract.image_to_string(Image.open(image_path), lang='lav', config=config)
        #text = pytesseract.image_to_string(Image.open(image_path), lang='lav')
        # cv2.IMREAD_UNCHANGED helps to read the image as it is, without any changes, especially useful for png with alpha channel
        text = pytesseract.image_to_string(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), lang=lang, config=config_w_blacklist)
        return text
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
        return None

def extract_filename_from_url(url):
    # Extracts the filename from the URL based on the expected pattern
    parts = url.split('/')
    # The pattern is based on your provided URL structure
    year, month, filename = parts[-3], parts[-2], parts[-1]
    local_filename = f"{year}_{month}_{filename}"
    return local_filename

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

def process_dataset(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                with open(json_path, 'r') as json_file:
                    data = json.load(json_file)
                    for item in data:
                        if 'main_image_url' in item:
                            expected_filename = extract_filename_from_url(item['main_image_url'])
                            image_path = os.path.join(root, expected_filename)
                            if os.path.exists(image_path):
                                lang = extract_lang(item)
                                extracted_text = run_ocr_on_image(image_path, lang)
                                print(f"Extracted text from {image_path}: {extracted_text}")
                            else:
                                print(f"Expected image file does not exist: {image_path}")


if __name__ == "__main__":
    dataset_directory = "dataset/timenote/Bolderajas_kapi/"
    process_dataset(dataset_directory)
    
