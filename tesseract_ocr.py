import cv2
import pytesseract

COMMON_BLACKLIST_CHARS = '0123456789.,/\\()[]}{#$%^&*!@~`-_=+<>?;:|'
LATIN_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

language_blacklists = {
    'lav': COMMON_BLACKLIST_CHARS + 'QWXYqwxy',
    'deu': COMMON_BLACKLIST_CHARS,                # no additional letters blacklisted
    'pol': COMMON_BLACKLIST_CHARS + 'QqXx',       # excluding 'Q', 'q', 'X', 'x'
    'rus': COMMON_BLACKLIST_CHARS + LATIN_LETTERS, # in case of mixed languages, excluding all Latin letters
}

def run_tesseract_ocr(image_path, lang='lav'):
    """
    Runs OCR on an image using Tesseract and returns the extracted text.
    :param image_path: Path to the image file.
    :param lang: Language code for Tesseract OCR (default is 'lav' for Latvian).
    :return: Extracted text or None if an error occurs.
    """
    try:
        blacklist_chars = language_blacklists.get(lang, COMMON_BLACKLIST_CHARS)
        config = f'--psm 3 --oem 3 -c tessedit_char_blacklist={blacklist_chars}'
        text = pytesseract.image_to_string(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), lang=lang, config=config)
        return text.strip()
    except Exception as e:
        print(f"Failed to process image {image_path} with Tesseract: {e}")
        return None
