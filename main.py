from similarity_metrics import basic_similarity_score, jaccard_similarity_score, levenshtein_similarity_allow_extras, combined_ngram_similarity_score
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
    print("Starting directory processing...")
    print("-" * 60)
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
                    print(f"  > Found JSON details for {file}")
                    # Extract true text using the image path to determine dataset type
                    true_text = get_true_text(json_details, image_path)
                    print(f"  > True text: {true_text}")
                else:
                    print(f"  > No JSON details found for {file}")
                ocr_text = run_tesseract_ocr(image_path, lang)
                if ocr_text:
                    #print(f"  > OCR text: {ocr_text[:50]}...")
                    print(f"  > OCR text: {ocr_text}...")
                else:
                    print("  > OCR text: [No text detected]")

                similarity_score = basic_similarity_score(ocr_text, true_text)
                print(f"  > Similarity score: {similarity_score}")

                jaccard_score = jaccard_similarity_score(ocr_text, true_text)
                print(f"  > Jaccard similarity score: {jaccard_score}")

                levenshtein_score = levenshtein_similarity_allow_extras(ocr_text, true_text)
                print(f"  > Levenshtein similarity score: {levenshtein_score}")

                ngram_score = combined_ngram_similarity_score(ocr_text, true_text)
                print(f"  > N-gram similarity score: {ngram_score}")

    print("-" * 60)
    print("Directory processing completed.")

    # TODO test similarity on synthetic data using ethalon texts with possible alterations

if __name__ == "__main__":
    #dataset_directory = "dataset/timenote/test"
    dataset_directory = "dataset/berlin-mitte/" 
    #dataset_directory = "dataset/timenote/"
    process_directory(dataset_directory)
