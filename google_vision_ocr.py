from google.cloud import vision
from google.oauth2 import service_account
import io
import os

class GoogleVisionOCR:
    def __init__(self):
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            raise EnvironmentError("Google Cloud credentials path not set in .env file")


    def perform_ocr(self, image_path):
        """Reads an image file and performs OCR using Google Vision API.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The extracted text from the image.
        """
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = self.client.text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(f'Google Vision API error: {response.error.message}')

        return texts[0].description if texts else ""
