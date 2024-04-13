import string
import unicodedata
import re
import jellyfish
from Levenshtein import distance as levenshtein_distance
from collections import Counter
from fuzzywuzzy import fuzz
from rapidfuzz import fuzz
import difflib

def normalize_text(text, strip_whitespace=False):
    """
    Normalize text by removing diacritics from characters, removing numbers and handling special cases.
    This includes converting German umlauts to their base letters and 'ß' to 'ss'.
    """
    text = text.replace('ß', 'ss')
    normalized = unicodedata.normalize('NFD', text)
    normalized = ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn' and not ch.isdigit())
    normalized = normalized.lower()  # Ensure lowercase
    if strip_whitespace:
        normalized = re.sub(r'\s+', '', normalized)  # Remove all whitespace
    else:
        normalized = re.sub(r'\s+', ' ', normalized)  # Standardize whitespace to single spaces
    return normalized

# word level similarity metrics

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

# char level similarity metrics

def levenshtein_similarity_allow_extras(true_text, ocr_text):
    """
    Calculates a modified Levenshtein similarity score that allows for extra characters in the OCR text.
    """
    true_text = normalize_text(true_text)
    ocr_text = normalize_text(ocr_text)

    min_distance = float('inf')
    true_text_len = len(true_text)

    if true_text_len == 0:
        return "0.00000"

    for i in range(len(ocr_text) - true_text_len + 1):
        segment = ocr_text[i:i+true_text_len]
        distance = levenshtein_distance(true_text, segment)
        min_distance = min(min_distance, distance)
    
    similarity_score = max(0, 1 - (min_distance / true_text_len))
    formatted_score = "{:.5f}".format(similarity_score)
    return formatted_score

def lcs_length(X, Y):
    """
    Computes the length of the longest common subsequence between two sequences.
    Dynamic programming approach.
    """
    m = len(X)
    n = len(Y)
    L = [[0] * (n+1) for i in range(m+1)]
  
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
  
    return L[m][n]

def lcs_similarity_score(true_text, ocr_text):
    """
    Calculates a similarity score based on the Longest Common Subsequence (LCS)
    between two normalized texts. The score is the length of the LCS divided by
    the length of the true text, to normalize the score.
    """
    true_text = normalize_text(true_text, True)
    ocr_text = normalize_text(ocr_text, True)
    
    # Preliminary check for any common character
    if not set(true_text).intersection(set(ocr_text)):
        return 0.0

    lcs_len = lcs_length(true_text, ocr_text)
    if len(true_text) == 0:
        return 0 if len(ocr_text) > 0 else 1  # Handling edge cases
    score = lcs_len / len(true_text)
    formatted_score = "{:.5f}".format(score)
    return formatted_score
    
def generate_ngrams(text, n=2):
    """
    Generate n-grams from the provided text after normalization.
    
    :param text: The text to generate n-grams from.
    :param n: The number of items in each n-gram.
    :return: A list of n-grams.
    """
    text = normalize_text(text)  # Apply normalization
    words = text.split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    return ngrams

def ngram_similarity_score(true_text, ocr_text, n=2):
    """
    Calculate the similarity between two texts based on their n-grams,
    with normalization applied, focusing on the presence of true text n-grams in the OCR text.
    
    :param true_text: The true text.
    :param ocr_text: The OCR-generated text.
    :param n: The number of items in each n-gram.
    :return: The similarity score as a float.
    """
    true_ngrams = generate_ngrams(true_text, n)
    ocr_ngrams = generate_ngrams(ocr_text, n)
    
    true_ngram_counts = Counter(true_ngrams)
    ocr_ngram_counts = Counter(ocr_ngrams)
    
    common_ngrams = set(true_ngrams) & set(ocr_ngrams)
    total_common_ngrams = sum(min(true_ngram_counts[ng], ocr_ngram_counts[ng]) for ng in common_ngrams)
    
    if not true_ngrams:
        return 1.0 if not ocr_ngrams else 0.0  # Handle edge cases
    similarity_score = total_common_ngrams / len(true_ngrams)
    
    return similarity_score

def combined_ngram_similarity_score(true_text, ocr_text):
    """
    Calculate a combined similarity score based on unigrams and bigrams.
    """
    # Unigram similarity
    unigram_similarity = ngram_similarity_score(true_text, ocr_text, n=1)
    
    # Bigram similarity
    bigram_similarity = ngram_similarity_score(true_text, ocr_text, n=2)
    
    # Average the unigram and bigram similarity scores
    combined_similarity = (unigram_similarity + bigram_similarity) / 2
    formatted_score = "{:.5f}".format(combined_similarity)
    return formatted_score

def jaro_winkler_similarity(true_text, ocr_text):
    """
    Calculate the Jaro-Winkler similarity between two texts after normalization.
    
    :param true_text: The true text.
    :param ocr_text: The OCR-generated text.
    :return: The similarity score as a float.
    """
    # Normalize texts
    normalized_true_text = normalize_text(true_text, True)
    normalized_ocr_text = normalize_text(ocr_text, True)

    # Calculate Jaro-Winkler similarity
    similarity = jellyfish.jaro_winkler_similarity(normalized_true_text, normalized_ocr_text)
    formatted_score = "{:.5f}".format(similarity)
    return formatted_score

def fuzzywuzzy_similarity(true_text, ocr_text):
    """
    Calculates the similarity score between two texts using the fuzzywuzzy library's ratio function.
    This function computes the Levenshtein distance percentage between two strings, providing a score
    between 0 and 100, where 100 represents a perfect match.

    :param true_text: The correct version of the text.
    :param ocr_text: The OCR-generated version of the text.
    :return: The similarity score as a percentage.
    """
    # Normalize texts
    normalized_true_text = normalize_text(true_text, True)
    normalized_ocr_text = normalize_text(ocr_text, True)

    # Calculate fuzzywuzzy similarity
    similarity = fuzz.ratio(normalized_true_text, normalized_ocr_text)
    return similarity / 100.0  # Convert to a score between 0 and 1

def rapidfuzz_similarity(true_text, ocr_text):
    """
    Calculates the similarity score between two texts using the RapidFuzz library's fuzz module.
    This function provides a similarity percentage based on advanced edit distances, which can
    offer better performance and accuracy for certain text comparison tasks.

    :param true_text: The correct version of the text.
    :param ocr_text: The OCR-generated version of the text.
    :return: The similarity score as a float between 0 and 1.
    """
    # Normalize texts
    normalized_true_text = normalize_text(true_text, True)
    normalized_ocr_text = normalize_text(ocr_text, True)

    # Calculate similarity using the rapidfuzz library
    similarity = fuzz.ratio(normalized_true_text, normalized_ocr_text)
    return similarity / 100.0  # Convert to a score between 0 and 1

def difflib_similarity(true_text, ocr_text):
    """
    Calculates the similarity score between two texts using the difflib library's SequenceMatcher.
    This method compares the sequences and outputs a similarity ratio, a float between 0 and 1,
    where 1 means the sequences are identical.

    :param true_text: The correct version of the text.
    :param ocr_text: The OCR-generated version of the text.
    :return: The similarity score as a float between 0 and 1.
    """
    # Normalize texts
    normalized_true_text = normalize_text(true_text, True)
    normalized_ocr_text = normalize_text(ocr_text, True)

    # Create a SequenceMatcher object and calculate the similarity ratio
    matcher = difflib.SequenceMatcher(None, normalized_true_text, normalized_ocr_text)
    similarity = matcher.ratio()
    return similarity