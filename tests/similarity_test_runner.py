import unittest
from test_levenshtein_similarity import TestLevenshteinSimilarity
from test_lcs_similarity import TestLCSSimilarity
from test_ngram_similarity import TestNGramSimilarity
from test_jaro_winkler_similarity import TestJaroWinklerSimilarity
from test_basic_similarity import TestBasicSimilarity
from test_jaccard_similarity import TestJaccardSimilarity

def create_test_suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestLevenshteinSimilarity))
    test_suite.addTest(unittest.makeSuite(TestLCSSimilarity))
    test_suite.addTest(unittest.makeSuite(TestNGramSimilarity))
    test_suite.addTest(unittest.makeSuite(TestJaroWinklerSimilarity))
    test_suite.addTest(unittest.makeSuite(TestBasicSimilarity))
    test_suite.addTest(unittest.makeSuite(TestJaccardSimilarity))
    return test_suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(create_test_suite())