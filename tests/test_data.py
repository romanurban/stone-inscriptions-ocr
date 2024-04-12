TEST_CASES = [
    ("Lorem Ipsum", "Lorem Ipsum", "ideal case"),
    ("Lorem Ipsum", "Lorme Ipsum", "typo"),
    ("Lorem Ipsum", "LoremIpsum", "no whitespace"),
    ("Lorem Ipsum dolor", "Lorem dolor", "missing tokens"),
    ("Lorem Ipsum", "Ipsum Lorem", "reordered words"),
    ("Lorem Ipsum", "d foo Lorem Ipsum dolor sit amet", "extra tokens"),
    ("The quick", "brown-fox", "completely different"),
    ("Lorem Ipsum", "", "OCR text empty")
]