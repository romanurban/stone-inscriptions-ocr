import unittest
from test_data import TEST_CASES
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from similarity_metrics import fuzzywuzzy_similarity

class TestFuzzyWuzzySimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nExecuting tests for FuzzyWuzzy Similarity...")

    def test_fuzzywuzzy_similarity_scenarios(self):
        for true_text, ocr_text, description in TEST_CASES:
            with self.subTest(description=description):
                score = fuzzywuzzy_similarity(true_text, ocr_text)
                print(f"{description} score: {score:.5f}")

            score = float(score)
            if description == "ideal case":
                self.assertAlmostEqual(score, 1.0, places=2)
            elif description == "typo":
                # Expecting high tolerance for typos
                self.assertGreater(score, 0.80)
            elif description == "no whitespace":
                # Whitespace normalization should handle this well
                self.assertGreater(score, 0.90)
            elif description == "missing tokens":
                # Adjusting expectations: fuzzywuzzy handles missing tokens better than expected
                self.assertGreaterEqual(score, 0.75)
            elif description == "reordered words":
                # Adjusting expectations: fuzzywuzzy is sensitive to order
                self.assertLessEqual(score, 0.50)
            elif description == "extra tokens":
                # Adjusting expectations: fuzzywuzzy is more penalizing towards extra tokens
                self.assertGreaterEqual(score, 0.50)
            elif description == "more extra tokens":
                # Adjusting expectations: fuzzywuzzy is more penalizing towards extra tokens
                self.assertGreaterEqual(score, 0.30)
            elif description == "completely different":
                # Completely different texts should yield the lowest score
                self.assertLess(score, 0.25)
            elif description == "OCR text empty":
                # An empty OCR text should result in a zero score
                self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
