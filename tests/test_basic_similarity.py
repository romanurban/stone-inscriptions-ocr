import unittest
from test_data import TEST_CASES  # Make sure this import is correct based on your project structure
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from similarity_metrics import basic_similarity_score


class TestBasicSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nExecuting tests for Basic Similarity...")

    def test_basic_similarity_scenarios(self):
        for true_text, ocr_text, description in TEST_CASES:
            with self.subTest(description=description):
                score = basic_similarity_score(ocr_text, true_text)  # Ensure parameters are in the correct order
                print(f"{description} score: {score}")

                score = float(score)
                if description == "ideal case":
                    self.assertEqual(score, 1.0)
                elif description == "typo":
                    # Typo case might not be directly applicable for basic similarity, adjust as needed
                    self.assertGreater(score, 0.0)
                elif description == "no whitespace":
                    # Basic similarity may not directly apply, but let's assume whitespace normalization
                    self.assertEqual(score, 0.0)
                elif description == "missing tokens":
                    # Expecting score to reflect missing tokens properly
                    self.assertLess(score, 1.0)
                elif description == "reordered words":
                    # Basic similarity typically doesn't consider order, so score might be high
                    self.assertGreater(score, 0.0)
                elif description == "extra tokens":
                    self.assertGreater(score, 0.0)
                elif description == "completely different":
                    self.assertEqual(score, 0.0)
                elif description == "OCR text empty":
                    self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
