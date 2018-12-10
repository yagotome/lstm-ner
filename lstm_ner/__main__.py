import os
import sys

from functools import reduce
from typing import Dict, List, Tuple

from keras.models import load_model
from keras.utils import np_utils

from lstm_ner.utils import data_utils
from lstm_ner import ner_model as ner

# defining constants
word_embeddings_file = 'data/word_embeddings_skipgram_wang2vec_50d.txt'
input_data_folder = 'data'
model_file = 'output/model.h5'
char_embeddings_file = 'output/char_embeddings.txt'

# defining hyper parameters
word_window_size = 5
char_window_size = 5
char_embeddings_dim = 20
dropout_rate = 0.5
lstm_units = 420
conv_num = 10
epochs = 20
test_percent = 0.2
not_entity_threshold = 0.7


class Metrics:
    def __init__(self):
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        self.actual_total = 0

    def total_predicted(self):
        return self.true_pos + self.true_neg + self.false_pos + self.false_neg

    def accuracy(self):
        return (self.true_pos + self.true_neg) / self.actual_total

    def precision(self):
        if self.true_pos + self.false_pos == 0:
            return 0
        return self.true_pos / (self.true_pos + self.false_pos)

    def recall(self):
        if self.true_pos + self.false_neg == 0:
            return 0
        return self.true_pos / (self.true_pos + self.false_neg)

    def f_measure(self):
        precision, recall = self.precision(), self.recall()
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)


def evaluate_model(predicted: List[Tuple[str, str]], actual: List[Tuple[str, str]], label2idx: Dict[str, int]):
    true_pos, true_neg, false_pos, false_neg = [0] * 4
    labeled_metrics: Dict[str, Metrics] = {label: Metrics() for label in label2idx.keys()}
    confusion_matrix: Dict[str, Dict[str, int]] = {actual_label: {pred_label: 0 for pred_label in label2idx.keys()} for
                                                   actual_label in label2idx.keys()}
    not_entity_label = 'O'
    for i, pred in enumerate(predicted):
        pred_label, actual_label = pred[1], actual[i][1]
        confusion_matrix[actual_label][pred_label] += 1
        labeled_metrics[actual_label].actual_total += 1
        if pred_label == actual_label == not_entity_label:
            true_neg += 1
            labeled_metrics[actual_label].true_neg += 1
        elif pred_label == not_entity_label:
            false_neg += 1
            labeled_metrics[actual_label].false_neg += 1
        elif pred_label == actual_label:
            true_pos += 1
            labeled_metrics[actual_label].true_pos += 1
        else:
            false_pos += 1
            labeled_metrics[pred_label].false_pos += 1
    print('TP: %d\nTN: %d\nFP: %d\nFN: %d' % (true_pos, true_neg, false_pos, false_neg))
    accuracy = (true_pos + true_neg) / len(predicted)
    print('Accuracy: %f' % accuracy)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f_measure = 2 * precision * recall / (precision + recall)

    print('Precision: %f\nRecall: %f\nF1 score: %f' % (precision, recall, f_measure))

    for label, metrics in labeled_metrics.items():
        print()
        print('==========>', label)
        print('TP: %d\nTN: %d\nFP: %d\nFN: %d' % (
            metrics.true_pos, metrics.true_neg, metrics.false_pos, metrics.false_neg))
        print('Total predicted: %f' % metrics.total_predicted())
        print('Total actual: %f' % metrics.actual_total)
        print('Accuracy: %f' % metrics.accuracy())
        print('Precision: %f' % metrics.precision())
        print('Recall: %f' % metrics.recall())
        print('F-measure: %f' % metrics.f_measure())
        print()

    # print_matrix
    print('Matriz de confusão (quantidades)')
    max_label = max(confusion_matrix.keys())
    print(f'{"".ljust(len(max_label)+3)}\t', end='')
    for label in confusion_matrix.keys():
        print(label, end='\t')
    print()
    for label in confusion_matrix.keys():
        print(label.ljust(3 + len(max_label)), end='\t')
        for amount in confusion_matrix[label].values():
            print(str(amount).ljust(5), end='\t')
        print()

    print()
    # print_matrix
    print('Matriz de confusão (percentual)')
    max_label = max(confusion_matrix.keys())
    print(f'{"".ljust(len(max_label)+3)}\t', end='')
    for label in confusion_matrix.keys():
        print(label, end='\t')
    print()
    for label in confusion_matrix.keys():
        print(label.ljust(3 + len(max_label)), end='\t')
        for amount in confusion_matrix[label].values():
            total = reduce(lambda a, b: a + b, confusion_matrix[label].values(), 0)
            if total == 0:
                amount, total = 0, 1
            print('%.2f%%' % (100 * amount / total), end='\t')
        print()

    return precision, recall, f_measure


def main():
    # getting args
    cpu_only = '--cpu-only' in sys.argv

    # loading data from files
    word_embeddings, word2idx, char2idx = data_utils.read_embeddings_file(word_embeddings_file)
    max_word_len = max(map(lambda word: len(word), word2idx.keys()))
    train_data, test_data, label2idx = data_utils.load_dataset(input_data_folder, test_percent)
    print('train sentences:', len(train_data))
    print('test sentences:', len(test_data))
    # train_data = train_data[:50]
    # test_data = test_data[:10]
    x_train, y_train = data_utils.transform_to_xy(train_data, word2idx, label2idx, word_window_size,
                                                  char2idx, max_word_len)
    x_test, y_test = data_utils.transform_to_xy(test_data, word2idx, label2idx, word_window_size,
                                                char2idx, max_word_len)
    num_labels = len(label2idx)
    # "binarize" labels
    y_train = np_utils.to_categorical(y_train, num_labels)
    y_test = np_utils.to_categorical(y_test, num_labels)
    # load model whether it is saved
    if os.path.exists(model_file):
        model = load_model(model_file)
        print(f'Model loaded from {model_file}')
        print(model.summary())
    else:
        # defining model
        word_input_length = 2 * word_window_size + 1
        max_word_len_padded = max_word_len + word_window_size * 2
        word_embedding_model = ner.generate_word_embedding_model(word_input_length, weights=word_embeddings)
        char_embedding_model = ner.generate_char_embedding_model(max_word_len, max_word_len_padded, word_input_length,
                                                                 char_embeddings_dim, conv_num, char_window_size,
                                                                 vocab_size=len(char2idx))
        model = ner.generate_model(word_embedding_model, char_embedding_model, lstm_units, num_labels, dropout_rate,
                                    cpu_only=cpu_only)

        # summarize the model
        print(model.summary())

        # training model
        model.fit(x_train, y_train, epochs=epochs)
        # model.fit([x_train[0]], y_train, epochs=epochs)

        # saving embeddings
        embedding_layer = char_embedding_model.layers[0]
        weights = embedding_layer.get_weights()[0]
        data_utils.save_embeddings(char_embeddings_file, weights, char2idx)

        # saving whole model
        model.save(model_file)

    # evaluating model
    loss, accuracy = model.evaluate(x_test, y_test)
    # loss, accuracy = model.evaluate([x_test[0]], y_test)
    print('Accuracy: %f' % (accuracy * 100))
    # output = model.predict([x_test[0][:15, :], x_test[1][:15, :]])
    output = model.predict(x_test)
    # output = model.predict([x_test[0]])

    train_data_flat = reduce(lambda acc, cur: acc + cur, train_data, [])
    label_dist = {label: 0 for label in label2idx.keys()}
    for word, label in train_data_flat:
        label_dist[label] += 1
    print()
    print('####### train label distribution')
    print('total: %d\n' % len(train_data_flat))
    for label, count in label_dist.items():
        print(label, count)
    print()

    test_data_flat = reduce(lambda acc, cur: acc + cur, test_data, [])
    labeled_output = label_output(output, label2idx, test_data_flat)
    evaluate_model(labeled_output, test_data_flat, label2idx)
    # precision, recall, f_measure = evaluate_model(labeled_output, test_data_flat, label2idx)
    # print('Precision: %f\nRecall: %f\nF1 score: %f' % (precision, recall, f_measure))

    # for word, ent in labeled_output:
    #     print(word + '\t' + ent)


def label_output(output: List[float], label2idx: Dict[str, int], test_data_flat: List[Tuple[str, str]]):
    classed_output = []
    for i in range(len(output)):
        not_entity_idx = label2idx['O']
        # if output[i, not_entity_idx] >= not_entity_threshold:
        #     entity = 'O'
        # else:
        ent_prob_max = 0
        ent_idx = not_entity_idx
        for j, ent in enumerate(output[i]):
            #     if ent > ent_prob_max and j != not_entity_idx:
            if ent > ent_prob_max:
                ent_prob_max = ent
                ent_idx = j
        entity = [label for label, idx in label2idx.items() if idx == ent_idx][0]
        classed_output.append((test_data_flat[i][0], entity))
    return classed_output


if __name__ == '__main__':
    main()
