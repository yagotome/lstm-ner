import re
from typing import List, Dict, Tuple

import numpy as np
from unidecode import unidecode


def read_input_file(filename: str):
    """
        Reads the input file and creates a list of sentences in which each sentence is a list of its word where the word
        is a 2-dim tuple, whose elements are the word itself and its label (named entity), respectively. Also creates
        a map of label to index.

        Expected files have a sequence of sentences. It has one word by line in first column (in a tab-separated file)
        followed in second column by its label, i.e., the named entity. The sentences are separated by an empty line.

        :param filename: Name of the file
        :return: List of sentences, map of label to index
    """
    sentences = []
    sentence = []
    label2idx = {'O': 0}
    label_idx = 1
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line == "":
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
                continue
            splits = line.split('\t')
            word = splits[0]
            label = splits[1]
            sentence.append((word, label))
            if label not in label2idx.keys():
                label2idx[label] = label_idx
                label_idx += 1
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences, label2idx


def tokenize_sentences(sentences: List[List[Tuple[str, str]]], word_indices: Dict[str, int],
                       label_indices: Dict[str, int]):
    unknown_idx = word_indices['UNKNOWN']

    def word2idx(word):
        if word in word_indices:
            return word_indices[word]
        lower = word.lower()
        if lower in word_indices:
            return word_indices[lower]
        normalized = normalize_word(word)
        if normalized in word_indices:
            return word_indices[normalized]
        return unknown_idx

    return [[(word2idx(word), label_indices[label]) for word, label in sentence] for sentence in sentences]


def multiple_replace(string, replace_dict):
    pattern = re.compile("|".join([re.escape(k) for k, v in replace_dict.items()]), re.M)
    return pattern.sub(lambda match: replace_dict[match.group(0)], string)


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


def create_context_windows(sentences: List[List[Tuple[int, int]]], window_size: int, padding_idx: int):
    """
    Generates X and Y matrices. X is an array of context window (indexed according to word2Idx). Each element of the
    array is the context window of the word in the middle and its index in the array is the index of its label in Y
    matrix.

    :param sentences: Sentences whose words and labels are already tokenized.
    :param window_size: How much words to the left and to the right.
    :param padding_idx: Index (token) for padding windows in which the main word has no enough surrounding words.
    :return: X and Y matrices as numpy array.
    """
    x_matrix = []
    y_vector = []
    for sentence in sentences:
        for target_word_idx in range(len(sentence)):
            word_indices = []
            for wordPosition in range(target_word_idx - window_size, target_word_idx + window_size + 1):
                if wordPosition < 0 or wordPosition >= len(sentence):
                    word_indices.append(padding_idx)
                    continue
                word_idx = sentence[wordPosition][0]
                word_indices.append(word_idx)
            label_idx = sentence[target_word_idx][1]
            x_matrix.append(word_indices)
            y_vector.append(label_idx)

    return np.array(x_matrix), np.array(y_vector)


def read_embeddings_file(filename: str):
    """
    Reads the embeddings file and maps its words to the index in the embeddings matrix

    :param filename: Name of the embeddings file
    :return: Embeddings matrix, map of word to index
    """
    word2idx = {}
    word_idx = 0
    char2idx = {'UNKNOWN': 0, 'PADDING': 1}
    char_idx = 2
    embeddings = []
    embeddings_dim = None
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            splits = line.strip().split(' ')
            if embeddings_dim is None:
                embeddings_dim = len(splits)
            else:
                assert embeddings_dim == len(splits)
            word = splits[0]
            for c in word:
                if c not in char2idx:
                    char2idx[c] = char_idx
                    char_idx += 1
            word2idx[word] = word_idx
            word_idx += 1
            embeddings.append(splits[1:])
    embeddings = np.array(embeddings, dtype=np.float32)
    return embeddings, word2idx, char2idx


def load_input_output_data(input_data_file: str, word2idx: Dict[str, int], word_window_size: int,
                           char2idx: Dict[str, int], char_window_size: int):
    sentences, label2idx = read_input_file(input_data_file)
    word_indexed_sentences = tokenize_sentences(sentences, word2idx, label2idx)
    char_indexed_sentences = tokenize_sentences(sentences, char2idx, label2idx)
    x_word, y = create_context_windows(word_indexed_sentences, word_window_size, word2idx['PADDING'])
    x_char, _ = create_context_windows(char_indexed_sentences, char_window_size, char2idx['PADDING'])
    x = [x_word, x_char]
    return x, y, label2idx
