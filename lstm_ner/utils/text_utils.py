import string
import nltk
import re
from typing import List, Tuple, Dict
from unidecode import unidecode


def remove_punctuations(text):
    translate_table = dict((ord(char), None) for char in string.punctuation)
    return text.translate(translate_table)


def generate_ngrams_freqdist(text, n):
    tokens = nltk.word_tokenize(text)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def multiple_replace(_string, replace_dict):
    pattern = re.compile("|".join([re.escape(k) for k, v in replace_dict.items()]), re.M)
    return pattern.sub(lambda match: replace_dict[match.group(0)], _string)


def normalize_word(line):
    """
    Transforms line to ASCII string making character translations, except some unicode characters are left because
    they are used in portuguese (such as ß, ä, ü, ö).
    """
    line = line.replace(u"„", u"\"")
    line = line.lower()

    replacements = dict(((u"ß", "SZ"), (u"ä", "AE"), (u"ü", "UE"), (u"ö", "OE")))
    replacements_inv = dict(zip(replacements.values(), replacements.keys()))
    line = multiple_replace(line, replacements)
    line = unidecode(line)
    line = multiple_replace(line, replacements_inv)

    line = line.lower()  # unidecode might have replaced some characters, like € to upper case EUR

    line = re.sub("([0-9][0-9.,]*)", '0', line)

    return line.strip()


def tokenize_sentences(sentences: List[List[Tuple[str, str]]], word_indices: Dict[str, int],
                       label_indices: Dict[str, int], char_level=False):
    unknown_idx = word_indices['UNKNOWN']

    def tokenize(_string):
        if _string in word_indices:
            return word_indices[_string]
        lower = _string.lower()
        if lower in word_indices:
            return word_indices[lower]
        normalized = normalize_word(_string)
        if normalized in word_indices:
            return word_indices[normalized]
        return unknown_idx

    def create_element(_string, label):
        if char_level:
            return [tokenize(c) for c in _string]
        return tokenize(_string), label_indices[label]

    return [[create_element(word, label) for word, label in sentence] for sentence in sentences]
