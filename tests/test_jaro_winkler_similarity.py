import unittest
from test_data import TEST_CASES
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from similarity_metrics import jaro_winkler_similarity


class TestJaroWinklerSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nExecuting tests for Jaro-Winkler Similarity...")

    def test_jaro_winkler_scenarios(self):
        for true_text, ocr_text, description in TEST_CASES:
            with self.subTest(description=description):
                score = jaro_winkler_similarity(true_text, ocr_text)
                print(f"{description} score: {score}")

                score = float(score)
                if description == "ideal case":
                    self.assertAlmostEqual(score, 1.0)
                elif description == "typo":
                    self.assertGreater(score, 0.8)
                elif description == "no whitespace":
                    self.assertGreater(score, 0.9)
                elif description == "missing tokens":
                    # The expected score can vary; adjust according to your similarity function's behavior
                    self.assertLess(score, 1.0)
                elif description == "reordered words":
                    # Jaro-Winkler is somewhat tolerant of reorderings, so expect a relatively high score
                    self.assertLess(score, 0.5)
                elif description == "extra tokens":
                    # Depending on how well your function handles extra tokens, the expected score might need adjustment
                    self.assertGreater(score, 0.7)
                elif description == "more extra tokens":
                    # Adjust the expected score based on how well your function handles extra tokens
                    self.assertGreater(score, 0.5)
                elif description == "completely different":
                    # Expect a low score for completely different texts
                    self.assertLess(score, 0.5)
                elif description == "OCR text empty":
                    # An empty OCR text should result in the lowest possible score
                    self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
