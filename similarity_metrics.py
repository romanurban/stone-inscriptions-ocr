import string
import unicodedata
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance

# TODO do not penalize for extra word tokens in OCR text, adjust metrics for that

def normalize_text(text):
    """
    Normalize text by removing diacritics from characters, removing numbers and handling special cases.
    This includes converting German umlauts to their base letters and 'ß' to 'ss'.
    """
    text = text.replace('ß', 'ss')
    normalized = unicodedata.normalize('NFD', text)
    normalized = ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn' and not ch.isdigit())
    normalized = normalized.lower()  # Ensure lowercase
    normalized = re.sub(r'\s+', ' ', normalized)  # Standardize whitespace
    return normalized

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

def cosine_similarity_tfidf_score(ocr_text, true_text):
    """
    Calculates the cosine similarity score between two texts using TF-IDF vectorization.
    This approach vectorizes the OCR and true text into TF-IDF vectors and then computes
    the cosine similarity between these vectors, providing a measure of textual similarity
    that considers both the frequency and significance of words.
    """
    ocr_text_norm = normalize_text(ocr_text)
    true_text_norm = normalize_text(true_text)
    
    # Vectorize the texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([ocr_text_norm, true_text_norm])
    
    # Calculate cosine similarity
    sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return "{:.5f}".format(sim_score)

def jaccard_similarity_score(ocr_text, true_text):
    """
    Calculates the Jaccard similarity score between two texts.
    The score is defined as the size of the intersection of the token sets
    of the two texts divided by the size of the union of the token sets of
    the two texts, reflecting how similar the texts are in terms of shared
    words regardless of their order or frequency.
    """
    # Normalize texts: lowercase and remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    norm_ocr_text = normalize_text(ocr_text.lower().translate(translator))
    norm_true_text = normalize_text(true_text.lower().translate(translator))
    
    # Tokenize texts into sets of words to calculate intersection and union
    ocr_words = set(norm_ocr_text.split())
    true_words = set(norm_true_text.split())
    
    # Calculate intersection and union
    intersection = len(ocr_words & true_words)
    union = len(ocr_words | true_words)
    
    # Avoid division by zero
    if union == 0:
        return "0.00000"
    
    # Calculate Jaccard similarity score
    jaccard_score = intersection / union
    formatted_score = "{:.5f}".format(jaccard_score)
    return formatted_score

def levenshtein_similarity_score(ocr_text, true_text):
    """
    Calculates a normalized Levenshtein similarity score between two texts.
    The score is normalized by the maximum possible distance (length of the longer text),
    so it ranges from 0 (completely different) to 1 (exactly the same).
    """
    # Normalize texts
    norm_ocr_text = normalize_text(ocr_text)
    norm_true_text = normalize_text(true_text)
    
    # Calculate Levenshtein distance
    distance = levenshtein_distance(norm_ocr_text, norm_true_text)
    
    # Normalize the distance to get a similarity score
    max_length = max(len(norm_ocr_text), len(norm_true_text))
    if max_length == 0:  # Prevent division by zero
        return "1.00000"  # Texts are the same if both are empty
    similarity = 1 - distance / max_length
    
    # Format similarity score up to 5 decimal places
    return "{:.5f}".format(similarity)
    
def generate_ngrams(text, n=2):
    """
    Generate n-grams from a given text.

    Parameters:
    - text (str): The text from which to generate n-grams.
    - n (int): The number of elements in each n-gram.

    Returns:
    - list: A list of n-gram strings.
    """
    # Remove spaces for character n-grams to treat the text as a continuous sequence of characters
    text = text.replace(" ", "").lower()
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    return ngrams

def ngram_similarity_score(ocr_text, true_text, n=2):
    """
    Calculate the similarity between two texts based on their n-grams.
    """
    ocr_ngrams = set(generate_ngrams(ocr_text, n))
    true_ngrams = set(generate_ngrams(true_text, n))

    # Calculate the Jaccard similarity
    intersection = len(ocr_ngrams.intersection(true_ngrams))
    union = len(ocr_ngrams.union(true_ngrams))
    if union == 0:
        return "1.00000"  # If both texts are empty, consider them identical
    jaccard_similarity = intersection / union
    
    # Format similarity score up to 5 decimal places
    return "{:.5f}".format(jaccard_similarity)