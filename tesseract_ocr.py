import cv2
import pytesseract
import numpy as np

class TesseractOCR:
    COMMON_BLACKLIST_CHARS = '0123456789.,/\\()[]}{#$%^&*!@~`-_=+<>?;:|'
    LATIN_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    def __init__(self):
        self.language_blacklists = {
            'lav': self.COMMON_BLACKLIST_CHARS + 'QWXYqwxy',
            'deu': self.COMMON_BLACKLIST_CHARS,                # No additional letters blacklisted
            'pol': self.COMMON_BLACKLIST_CHARS + 'QqXx',       # Excluding 'Q', 'q', 'X', 'x'
            'rus': self.COMMON_BLACKLIST_CHARS + self.LATIN_LETTERS, # Excluding all Latin letters
        }

    def run_ocr(self, image_input, lang='lav'):
        """
        Runs OCR on an image using Tesseract with the specified language and returns the extracted text.
        :param image_path: Path to the image file.
        :param lang: Language code for Tesseract OCR.
        :return: Extracted text or None if an error occurs.
        """
        try:
            # Check if the input is a numpy array
            if isinstance(image_input, np.ndarray):
                image = image_input
            else:
                # If the input is not a numpy array, assume it's a file path
                image = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)

            blacklist_chars = self.language_blacklists.get(lang, self.COMMON_BLACKLIST_CHARS)
            config = f'--psm 3 --oem 3 -c tessedit_char_blacklist={blacklist_chars}'
            text = pytesseract.image_to_string(image, lang=lang, config=config)
            return text.strip()
        except Exception as e:
            print(f"Failed to process image with Tesseract in language '{lang}': {e}")
            return None
