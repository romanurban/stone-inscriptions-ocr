import json
import os
from similarity_metrics import (
    basic_similarity_score, lcs_similarity_score, jaro_winkler_similarity, difflib_similarity
)
from composite_score_calculator import CompositeScoreCalculator

class ScoreService:
    def __init__(self, revision):
        self.base_directory = os.path.join(f"ocr_results/revision_{revision}")
        self.ensure_directory(self.base_directory)

    def ensure_directory(self, path):
        """Ensure that the directory exists."""
        os.makedirs(path, exist_ok=True)

    def process_scores(self, full_file_path, ocr_method, true_text, ocr_text):
        """Processes the OCR scores and logs them based on specified parameters."""
        # if not true_text or not ocr_text:
        #     print(f"No text to process for file {full_file_path}. Skipping scoring.")
        #     return

        scores = {
            'basic_similarity_score': basic_similarity_score(ocr_text, true_text),
            'lcs_similarity_score': lcs_similarity_score(ocr_text, true_text),
            'jaro_winkler_similarity': jaro_winkler_similarity(ocr_text, true_text),
            'difflib_similarity': difflib_similarity(ocr_text, true_text)
        }

        selected_scores = [
            scores['lcs_similarity_score'],
            scores['jaro_winkler_similarity'],
            scores['basic_similarity_score'],
            scores['difflib_similarity']
        ]

        composite_score = CompositeScoreCalculator(selected_scores).calculate()
        score_entry = {
            'file_id': full_file_path,
            'ocr_method': ocr_method,
            'true_text': true_text,
            'ocr_text': ocr_text,
            'scores': scores,
            'composite_score': composite_score
        }

        # Log scores to directory-specific file
        self._log_scores(full_file_path, score_entry)

    def _log_scores(self, full_file_path, score_entry):
        """Logs the score data dynamically based on the full file path."""
        output_directory = os.path.join(self.base_directory, os.path.dirname(full_file_path).strip("./"))
        output_file = os.path.join(output_directory, 'scores.json')
        self.ensure_directory(output_directory)

        with open(output_file, 'a') as f:
            json_entry = json.dumps(score_entry, ensure_ascii=False, indent=4)
            f.write(json_entry + ",\n")  # Append as a new JSON object
        print(json_entry)
