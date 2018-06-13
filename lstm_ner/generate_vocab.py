import glob
from typing import List

from lstm_ner.utils import text_utils

input_path = 'input'
n = 7
score_delta = 5
threshold = 0.01

if __name__ == '__main__':
    assert n > 0
    text = ''
    for filename in glob.glob(f'{input_path}/*.txt'):
        with open(filename, 'r', encoding='utf-8') as file:
            text += '\n' + file.read()
    text = text_utils.remove_punctuations(text).strip()
    unigrams, tokens = text_utils.generate_ngrams_freqdist(text, 0)
    word_list = list(map(lambda k: k[0], unigrams.keys()))
    ngrams_list = [unigrams]
    for i in range(1, n):
        ngrams, _ = text_utils.generate_ngrams_freqdist(text, i + 1, tokens)
        ngrams_list.append(ngrams)

    # generate sequence of words in which ngrams whose score is gte threshold are represented as a single word
    initial_word_sequence = text.split()
    sequence = []
    i = 0
    ngrams_count = 0
    while i < len(initial_word_sequence):
        start_idx = i
        ngram_to_score: List[str]
        end_idx: int
        for j in range(n - 1, -1, -1):
            end_idx = i + j
            ngram_to_score = word_list[start_idx: end_idx + 1]
            # TODO: optimize the following function so that processing is faster
            score = text_utils.score_ngrams(ngram_to_score, ngrams_list[j], unigrams, score_delta)
            # print('"' + ' '.join(ngram_to_score) + '"', 'has score:', score)
            if score > threshold:
                if j > 0:
                    ngrams_count += 1
                break
        sequence.append(tuple(ngram_to_score))
        i = end_idx + 1
    print(ngrams_count)
