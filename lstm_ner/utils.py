def read_input_file(filename: str):
    """
        Reads the input file and creates a list of sentences in which each sentence is a list of its word where the word
        is a 2-dim tuple, whose elements are the word itself and its label (named entity), respectively.

        Expected files have a sequence of sentences. It has one word by line in first column (in a tab-separated file)
        followed in second column by its label, i.e., the named entity. The sentences are separated by an empty line.

        :param filename: Name of the file.
    """
    sentences = []
    sentence = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == "":
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
                continue
            splits = line.split('\t')
            sentence.append((splits[0], splits[1]))
    return sentences
