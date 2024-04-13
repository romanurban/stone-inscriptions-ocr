import cv2
import pytesseract

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

    def run_ocr(self, image_path, lang='lav'):
        """
        Runs OCR on an image using Tesseract with the specified language and returns the extracted text.
        :param image_path: Path to the image file.
        :param lang: Language code for Tesseract OCR.
        :return: Extracted text or None if an error occurs.
        """
        try:
            blacklist_chars = self.language_blacklists.get(lang, self.COMMON_BLACKLIST_CHARS)
            config = f'--psm 3 --oem 3 -c tessedit_char_blacklist={blacklist_chars}'
            text = pytesseract.image_to_string(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), lang=lang, config=config)
            return text.strip()
        except Exception as e:
            print(f"Failed to process image {image_path} with Tesseract in language '{lang}': {e}")
            return None
