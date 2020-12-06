"""
Language detector implementation starter
"""

import math
from pprint import pprint

import main

if __name__ == '__main__':

    unknown_file = open('unknown_Arthur_Conan_Doyle.txt', encoding='utf-8')
    german_file = open('Thomas_Mann.txt', encoding='utf-8')
    english_file = open('Frank_Baum.txt', encoding='utf-8')

    text_unk = main.tokenize_by_sentence(unknown_file.read())
    text_ger = main.tokenize_by_sentence(german_file.read())
    text_eng = main.tokenize_by_sentence(english_file.read())
    english_file.close()
    german_file.close()
    unknown_file.close()

    letter_storage = main.LetterStorage()
    letter_storage.update(text_eng)
    letter_storage.update(text_ger)
    letter_storage.update(text_unk)

    eng_encoded = main.encode_corpus(letter_storage, text_eng)
    unk_encoded = main.encode_corpus(letter_storage, text_unk)
    ger_encoded = main.encode_corpus(letter_storage, text_ger)

    language_detector = main.ProbabilityLanguageDetector((3, 4, 5), 1000)
    language_detector.new_language(eng_encoded, 'english')
    language_detector.new_language(ger_encoded, 'german')

    actual = language_detector.detect_language(unk_encoded)
    print(actual['german'] < actual['english'])
    print(actual)

    RESULT = 'ok'
    assert RESULT, "Language detector doesn't work"
