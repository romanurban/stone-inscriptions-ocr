import unittest
from test_data import TEST_CASES
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from similarity_metrics import lcs_similarity_score

class TestLCSSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nExecuting tests for LCS Similarity...")

    def test_lcs_similarity_scenarios(self):
        for true_text, ocr_text, description in TEST_CASES:
            with self.subTest(description=description):
                score = lcs_similarity_score(true_text, ocr_text)
                print(f"{description} score: {score}")

                score = float(score)
                if description == "ideal case":
                    self.assertEqual(score, 1.0)
                elif description == "typo":
                    # LCS should handle typos relatively well, depending on their impact on the longest common subsequence
                    self.assertGreater(score, 0.5)
                elif description == "no whitespace":
                    # Assuming normalization handles whitespace, LCS might not be as affected, but it's less flexible than Jaro-Winkler
                    self.assertGreaterEqual(score, 0.8)
                elif description == "missing tokens":
                    # Missing tokens directly affect the LCS length, thus reducing the score
                    self.assertGreater(score, 0.0)
                    self.assertLess(score, 1.0)
                elif description == "reordered words":
                    # Reordering affects LCS since it relies on sequence order; score depends on the extent of reordering
                    self.assertLess(score, 1.0)
                elif description == "extra tokens":
                    # Extra tokens can potentially increase the length of the sequence without affecting the LCS of the true text
                    self.assertGreaterEqual(score, 1.0)
                elif description == "more extra tokens":
                    # More extra tokens can increase the length of the sequence without affecting the LCS of the true text
                    self.assertGreaterEqual(score, 1.0)
                elif description == "completely different":
                    # Completely different texts should have a very low LCS score
                    self.assertLess(score, 0.5)
                elif description == "OCR text empty":
                    # An empty OCR text means no common subsequence
                    self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
