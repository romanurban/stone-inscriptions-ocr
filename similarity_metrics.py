import string
import unicodedata

def normalize_text(text):
    """
    Normalize text by removing diacritics from characters and handling special cases.
    This includes converting German umlauts to their base letters and 'ß' to 'ss'.
    """
    # Specific replacement for ß to 'ss' before normalization
    text = text.replace('ß', 'ss')
    
    # Decompose the remaining characters and remove diacritic marks
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn')

def basic_similarity_score(ocr_text, true_text):
    """
    Compares OCR text with true text in a naive way by checking the proportion
    of matching words, disregarding the order and context.
    """
    # Normalize texts: lowercase and remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    norm_ocr_text = normalize_text(ocr_text.lower().translate(translator))
    norm_true_text = normalize_text(true_text.lower().translate(translator))
    
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
