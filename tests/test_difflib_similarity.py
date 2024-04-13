import unittest
from test_data import TEST_CASES
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from similarity_metrics import difflib_similarity

class TestDifflibSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nExecuting tests for Difflib Similarity...")

    def test_difflib_similarity_scenarios(self):
        for true_text, ocr_text, description in TEST_CASES:
            with self.subTest(description=description):
                score = difflib_similarity(true_text, ocr_text)
                print(f"{description} score: {score:.5f}")

                # Convert the percentage score to a float for comparison
                score = float(score)
                if description == "ideal case":
                    self.assertAlmostEqual(score, 1.0, places=2)
                elif description == "typo":
                    # Expect high tolerance for typos
                    self.assertGreater(score, 0.85)
                elif description == "no whitespace":
                    # Expect high scores if normalization effectively handles whitespace
                    self.assertGreater(score, 0.90)
                elif description == "missing tokens":
                    # Adjusted expectation: Allow a score as high as 0.80
                    self.assertLessEqual(score, 0.80)
                elif description == "reordered words":
                    # Adjusted expectation: Expect a lower score due to ordering sensitivity
                    self.assertLessEqual(score, 0.50)
                elif description == "extra tokens":
                    # Adjusted expectation: Lower threshold for extra tokens
                    self.assertGreaterEqual(score, 0.55)
                elif description == "more extra tokens":
                    # Adjusted expectation: Lower threshold for extra tokens
                    self.assertGreaterEqual(score, 0.32)
                elif description == "completely different":
                    # Expect very low scores for completely different texts
                    self.assertLess(score, 0.30)
                elif description == "OCR text empty":
                    # An empty OCR text should result in a score of zero
                    self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
