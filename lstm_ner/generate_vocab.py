import glob
from lstm_ner.utils import text_utils

input_path = 'input'
n = 7
score_delta = 5
threshold = 0.01

if __name__ == '__main__':
    ngrams_list = []
    word_list = None
    for filename in glob.glob(f'{input_path}/*.txt'):
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
            text = text_utils.remove_punctuations(text).strip()
            assert n > 0
            unigrams, tokens = text_utils.generate_ngrams_freqdist(text, 0)
            ngrams_list.append(unigrams)
            word_list = list(map(lambda k: k[0], ngrams_list[0].keys()))
            for i in range(1, n):
                ngrams, _ = text_utils.generate_ngrams_freqdist(text, i + 1, tokens)
                ngrams_list.append(ngrams)

                start_idx = word_list.index('Cândido')
                ngram_to_score = word_list[start_idx: start_idx + i + 1]

                score = text_utils.score_ngrams(ngram_to_score, ngrams_list[i], unigrams, score_delta)
                print(score, '"' + ' '.join(ngram_to_score) + '"', 'is', ('not' if score > threshold else '\b'),
                      'a unique name')
