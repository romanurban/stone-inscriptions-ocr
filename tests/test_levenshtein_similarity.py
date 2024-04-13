import unittest
from test_data import TEST_CASES
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from similarity_metrics import levenshtein_similarity_allow_extras

class TestLevenshteinSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nExecuting tests for Levenshtein Similarity...")

    def test_levenshtein_similarity_scenarios(self):
        for true_text, ocr_text, description in TEST_CASES:
            with self.subTest(description=description):
                score = levenshtein_similarity_allow_extras(true_text, ocr_text)
                print(f"{description} score: {score}")

                score = float(score)
                if description == "ideal case":
                    self.assertAlmostEqual(score, 1.0)
                elif description == "typo":
                    # Levenshtein should be relatively forgiving for a single typo
                    self.assertGreater(score, 0.8)
                elif description == "no whitespace":
                    # Depending on normalization, this might be close to ideal
                    self.assertLess(score, 0.5)
                elif description == "missing tokens":
                    # Missing tokens should decrease the score, but it may still be significant
                    self.assertEqual(score, 0.0)
                elif description == "reordered words":
                    # Reordering significantly affects Levenshtein, expect a lower score
                    self.assertLess(score, 0.8)
                elif description == "extra tokens":
                    # Extra tokens will affect the score, but it should be relatively high if all true text tokens are found
                    self.assertGreater(score, 0.5)
                elif description == "more extra tokens":
                    # More extra tokens will affect the score, but it should be relatively high if all true text tokens are found
                    self.assertGreater(score, 0.5)
                elif description == "completely different":
                    # Completely different texts should yield the lowest score
                    self.assertEqual(score, 0.0)
                elif description == "OCR text empty":
                    # An empty OCR text means no similarity
                    self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
