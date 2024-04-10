import cv2
import pytesseract

def run_tesseract_ocr(image_path, lang='lav'):
    """
    Runs OCR on an image using Tesseract and returns the extracted text.
    :param image_path: Path to the image file.
    :param lang: Language code for Tesseract OCR (default is 'lav' for Latvian).
    :return: Extracted text or None if an error occurs.
    """
    try:
        blacklist_chars = '0123456789.,/\\()[]}{#$%^&*!@~`-_=+<>?;:|'
        blacklist_letters_lv = 'QWXYqwxy'
        blacklist_lv = blacklist_chars + blacklist_letters_lv
        config = f'--psm 3 --oem 3 -c tessedit_char_blacklist={blacklist_lv}'
        text = pytesseract.image_to_string(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), lang=lang, config=config)
        return text.strip()
    except Exception as e:
        print(f"Failed to process image {image_path} with Tesseract: {e}")
        return None
