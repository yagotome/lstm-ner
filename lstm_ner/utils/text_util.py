import string
import nltk


def remove_punctuations(text):
    translate_table = dict((ord(char), None) for char in string.punctuation)
    return text.translate(translate_table)


def generate_ngrams_freqdist(text, n):
    tokens = nltk.word_tokenize(text)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)
