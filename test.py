import main

def test_probability_language_detector_detect_language_ideal():
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

    print('german', actual['german'], 'english', actual['english'])
    print(actual['german'] > actual['english'])

def test_probability_language_detector_calculate_probability_ideal():
    english_file = open('Frank_Baum.txt', encoding='utf-8')
    german_file = open('Thomas_Mann.txt', encoding='utf-8')
    unknown_file = open('unknown_Arthur_Conan_Doyle.txt', encoding='utf-8')

    english_text = main.tokenize_by_sentence(english_file.read())
    german_text = main.tokenize_by_sentence(german_file.read())
    unknown_text = main.tokenize_by_sentence(unknown_file.read())

    english_file.close()
    german_file.close()
    unknown_file.close()

    letter_storage = main.LetterStorage()
    letter_storage.update(english_text)
    letter_storage.update(german_text)
    letter_storage.update(unknown_text)

    english_encoded = main.encode_corpus(letter_storage, english_text)
    german_encoded = main.encode_corpus(letter_storage, german_text)
    unknown_encoded = main.encode_corpus(letter_storage, unknown_text)

    language_detector = main.ProbabilityLanguageDetector((3,), 1000)
    language_detector.new_language(english_encoded, 'english')
    language_detector.new_language(german_encoded, 'german')

    n3_gram_trie_english = language_detector.n_gram_storages['english'][3]
    n3_gram_trie_german = language_detector.n_gram_storages['german'][3]

    n3_gram_unknown = main.NGramTrie(3)
    n3_gram_unknown.fill_n_grams(unknown_encoded)

    english_prob = language_detector._calculate_sentence_probability(n3_gram_trie_english,
                                                                     n3_gram_unknown.n_grams)
    german_prob = language_detector._calculate_sentence_probability(n3_gram_trie_german,
                                                                    n3_gram_unknown.n_grams)
    print(f'English_sentence_prob: {english_prob}')
    print(f'Deutsch_sentence_prob: {german_prob}')
    print(english_prob > german_prob)

if __name__ == "__main__":
    #test_probability_language_detector_detect_language_ideal()
    test_probability_language_detector_calculate_probability_ideal()