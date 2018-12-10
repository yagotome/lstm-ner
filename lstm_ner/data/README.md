# Data directory

This directory must contain the data used to train and test the model.


## Data types

There are two kind of input data for this project, _TXT_ and _TSV_ files.


### Word Embeddings file

This file is a _TXT_ containing pre-trained word vector for each word in the _corpus_.


#### Where to download

You can find some pre-trained word embeddings file in the following links:

- [English](https://nlp.stanford.edu/projects/glove/)
- [Portuguese](http://nilc.icmc.usp.br/embeddings)


#### Preparing word embeddings file

The files from that websites is almost ready to be used, except that their first line is a metadata and it does not have the special tokens _PADDING_ and _UNKOWN_ that are necessary to represent a _context window padding_ and a _unkown word/token_ (word not in the corpus), respectively.

So there is a [script](prepare_data.py) to prepare files like those. The script does nothing else than remove _metadata line_ (first line) and add both _PADDING_ and _UNKOWN_ tokens at the begining of the file. To run it, just type:

```
$ python prepare_data.py <path/to/file.txt> <embedding_dimension>
```

You must adjust in [main](../__main__.py) that file name in `word_embeddings_file` variable.


### Training/Test input data

These files are the ones used to train and test the Neural Network. They are _TSV_ files in [_CoreNLP_ format](https://stanfordnlp.github.io/CoreNLP/ner.html#training-or-retraining-new-models).

This is a sample of those files:

```
Joe    PERSON
Smith  PERSON
lives  O
in     O
California    LOCATION
.    O

He    O
used    O
to    O
live    O
in    O
Oregon    LOCATION
.    O
```

You can add to data folder as much _TSV_ files as you want. All of them are used, and the training and test set separation is done in code.
