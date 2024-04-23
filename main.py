import os
from dotenv import load_dotenv
from tesseract_ocr import TesseractOCR
from google_vision_ocr import GoogleVisionOCR
from similarity_score_service import ScoreService
from dataset_helper import get_true_text, get_json_details, extract_lang

REVISION = "INITIAL"
DEFAULT_LANGUAGE = 'lav' # default latvian language code for timenote dataset
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
MITTE_DS_LANG_CODE = 'deu' # default german language code for the 'berlin-mitte/' dataset

def process_directory(directory):
    lang = DEFAULT_LANGUAGE
    print("Starting directory processing...")
    print("-" * 60)
    google_vision_ocr = GoogleVisionOCR()
    tesseract_ocr = TesseractOCR() 
    score_service = ScoreService(REVISION)  # Set a base directory for scores

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                image_path = os.path.join(root, file)
                true_text = ""  # Default value if no JSON details are found or if an error occurs
                print("\nProcessing image:", image_path)
                json_details = get_json_details(image_path, root)
                if json_details:
                    if 'timenote' in image_path:
                        lang = extract_lang(json_details)
                    else:
                        lang = MITTE_DS_LANG_CODE
                    true_text = get_true_text(json_details, image_path)  # Extract true text
                    print(f"  > True text: {true_text}")
                else:
                    print(f"  > No JSON details found for {file}")

                ocr_text = tesseract_ocr.run_ocr(image_path, lang)
                google_vision_ocr_text = google_vision_ocr.perform_ocr(image_path)

                print(f"  > Tesseract OCR text: {ocr_text if ocr_text else '[No text detected]'}")
                print(f"  > Google Vision OCR text: {google_vision_ocr_text if google_vision_ocr_text else '[No text detected]'}")
                # TODO: add apple vision ocr

                score_service.process_scores(image_path, "Tesseract", true_text, ocr_text)
                score_service.process_scores(image_path, "Google Vision", true_text, google_vision_ocr_text)

    print("-" * 60)
    print("Directory processing completed.")

if __name__ == "__main__":
    #dataset_directory = "dataset/timenote/"

    #dataset_directory = "dataset/timenote/Bikernieku_kapi"
    #dataset_directory = "dataset/timenote/Bolderajas_kapi"
    #dataset_directory = "dataset/timenote/Jaunciema_kapi"
    #dataset_directory = "dataset/timenote/Katlakalna_kapi"
    #dataset_directory = "dataset/timenote/Lacupes_kapi"
    #dataset_directory = "dataset/timenote/Pleskodales_kapi"
    #dataset_directory = "dataset/timenote/Tornakalna_kapi"
    
    #dataset_directory = "dataset/timenote/test"
    dataset_directory = "dataset/berlin-mitte/" 
    process_directory(dataset_directory)