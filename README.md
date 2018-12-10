# LSTM NER

An implementation of Named Entity Recognition using LSTM Networks in Keras


## Setup

### Prerequisite

- python **3.6** (due to [`tensorflow-gpu` dependency](https://github.com/tensorflow/tensorflow/issues/8251))
- pip3 v18


### Setting up enviroment with Anaconda

#### Prerequisite

- anaconda3 v5 installed and set in _PATH_


#### Setup

```
$ conda create -n lstm-ner python=3.6
$ conda activate lstm-ner
$ pip install -r requirements-(cpu|gpu).txt
```


### Setting up data to train

See [data README](lstm_ner/data/README.md)


## Running

In `lstm_ner` folder, type:

```
$ PYTHONPATH=<path/to/lstm-ner> python __main__.py [--cpu-only]
```

PS: Set `PYTHONPATH` variable to `lstm-ner` root directory.