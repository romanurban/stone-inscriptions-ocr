from google.cloud import vision
from google.oauth2 import service_account
import io
import os
from PIL import Image
import io
from dotenv import load_dotenv
import numpy as np

load_dotenv()
creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

class GoogleVisionOCR:
    def __init__(self):
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            raise EnvironmentError("Google Cloud credentials path not set in .env file")


    def perform_ocr(self, image_input):
        """Reads an image file and performs OCR using Google Vision API.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The extracted text from the image.
        """
        if isinstance(image_input, np.ndarray):
            # Convert the numpy array to a PIL Image
            pil_image = Image.fromarray(image_input)

            # Save the PIL Image to a bytes buffer in PNG format
            byte_buffer = io.BytesIO()
            pil_image.save(byte_buffer, format='PNG')

            # Get the bytes value from the buffer
            content = byte_buffer.getvalue()
        else:
            with io.open(image_input, 'rb') as image_file:
                content = image_file.read()

        image = vision.Image(content=content)
        response = self.client.text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(f'Google Vision API error: {response.error.message}')

        return texts[0].description if texts else ""
