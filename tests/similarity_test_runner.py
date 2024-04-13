import unittest
from test_levenshtein_similarity import TestLevenshteinSimilarity
from test_lcs_similarity import TestLCSSimilarity
from test_ngram_similarity import TestNGramSimilarity
from test_jaro_winkler_similarity import TestJaroWinklerSimilarity
from test_basic_similarity import TestBasicSimilarity
from test_jaccard_similarity import TestJaccardSimilarity
from test_fuzzywuzzy_similarity import TestFuzzyWuzzySimilarity
from test_rapidfuzz_similarity import TestRapidFuzzSimilarity
from test_difflib_similarity import TestDifflibSimilarity

def create_test_suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestLevenshteinSimilarity))
    test_suite.addTest(unittest.makeSuite(TestLCSSimilarity))
    test_suite.addTest(unittest.makeSuite(TestNGramSimilarity))
    test_suite.addTest(unittest.makeSuite(TestJaroWinklerSimilarity))
    test_suite.addTest(unittest.makeSuite(TestBasicSimilarity))
    test_suite.addTest(unittest.makeSuite(TestJaccardSimilarity))
    test_suite.addTest(unittest.makeSuite(TestFuzzyWuzzySimilarity))
    test_suite.addTest(unittest.makeSuite(TestRapidFuzzSimilarity))
    test_suite.addTest(unittest.makeSuite(TestDifflibSimilarity))
    return test_suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(create_test_suite())