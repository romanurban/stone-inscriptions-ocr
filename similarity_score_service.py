import json
import os
from similarity_metrics import (
    basic_similarity_score, lcs_similarity_score, jaro_winkler_similarity, difflib_similarity
)
from composite_score_calculator import CompositeScoreCalculator

class ScoreService:
    def __init__(self, basedir, revision):
        self.base_directory = os.path.join("ocr_results", basedir)
        self.output_file = os.path.join(self.base_directory, f"scores_revision_{revision}.json")
        self.ensure_directory(self.base_directory)

    def ensure_directory(self, path):
        """Ensure that the directory exists."""
        os.makedirs(path, exist_ok=True)

    def process_scores(self, full_file_path, ocr_method, true_text, ocr_text):
        """Processes the OCR scores and logs them based on specified parameters."""
        if not true_text or not ocr_text:
            print(f"No text to process for file {full_file_path}. Skipping scoring.")
            return

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
        score_data = {
            'scores': scores,
            'composite_score': composite_score
        }

        self._log_scores(full_file_path, ocr_method, score_data, to_file=True)

    def _log_scores(self, full_file_path, ocr_method, score_data, to_file):
        """Logs the score data to the console or writes to a file based on the 'to_file' parameter."""
        entry = {
            'file_id': full_file_path,
            'ocr_method': ocr_method,
            'score_data': score_data
        }
        if to_file and self.output_file:
            with open(self.output_file, 'a') as f:
                json.dump(entry, f, ensure_ascii=False, indent=4)
                f.write("\n")

        print(f"Scores for {full_file_path} [{ocr_method}]: {json.dumps(entry, ensure_ascii=False, indent=4)}")
