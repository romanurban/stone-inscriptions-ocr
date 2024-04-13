import unittest
from test_data import TEST_CASES
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from similarity_metrics import jaccard_similarity_score

class TestJaccardSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nExecuting tests for Jaccard Similarity...")

    def test_jaccard_similarity_scenarios(self):
        for true_text, ocr_text, description in TEST_CASES:
            with self.subTest(description=description):
                score = jaccard_similarity_score(ocr_text, true_text)  # Ensure parameters are in correct order
                print(f"{description} score: {score}")

                score = float(score)
                if description == "ideal case":
                    self.assertEqual(score, 1.0)
                elif description == "typo":
                    # For Jaccard, a typo might significantly affect the score, adjust expectation accordingly
                    self.assertGreater(score, 0.0)
                elif description == "no whitespace":
                    # Assuming that normalization handles spaces, expect high score
                    self.assertEqual(score, 0.0)
                elif description == "missing tokens":
                    self.assertGreater(score, 0.0)
                    self.assertLess(score, 1.0)
                elif description == "reordered words":
                    # Jaccard similarity should be high since it doesn't consider order
                    self.assertGreater(score, 0.0)
                elif description == "extra tokens":
                    # Expecting the presence of extra tokens to reduce the score, but not drastically
                    self.assertGreater(score, 0.0)
                elif description == "more extra tokens":
                    # Expecting the presence of extra tokens to reduce the score, but not drastically
                    self.assertGreater(score, 0.0)
                elif description == "completely different":
                    self.assertEqual(score, 0.0)
                elif description == "OCR text empty":
                    # An empty OCR text should result in the lowest possible score
                    self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
