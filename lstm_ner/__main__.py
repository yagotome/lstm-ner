import os

from functools import reduce

from keras.models import load_model
from keras.utils import np_utils

from lstm_ner.utils import data_utils
from lstm_ner import ner_model as ner

# defining constants
word_embeddings_file = 'data/word_embeddings.txt'
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
epochs = 50
test_percent = 0.2
not_entity_threshold = 0.7


def main():
    # loading data from files
    word_embeddings, word2idx, char2idx = data_utils.read_embeddings_file(word_embeddings_file)
    max_word_len = max(map(lambda word: len(word), word2idx.keys()))
    train_data, test_data, label2idx = data_utils.load_dataset(input_data_folder, test_percent)
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
        model = ner.generate_model(word_embedding_model, char_embedding_model, lstm_units, num_labels, dropout_rate)

        # summarize the model
        print(model.summary())

        # training model
        model.fit(x_train, y_train, epochs=epochs)

        # saving embeddings
        embedding_layer = char_embedding_model.layers[0]
        weights = embedding_layer.get_weights()[0]
        data_utils.save_embeddings(char_embeddings_file, weights, char2idx)

        # saving whole model
        model.save(model_file)
    # evaluating model
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Accuracy: %f' % (accuracy * 100))
    output = model.predict(x_test)
    labeled_output = label_output(output, label2idx, test_data)
    for word, ent in labeled_output:
        print(word + '\t' + ent)


def label_output(output, label2idx, test_data):
    classed_output = []
    test_data_flat = reduce(lambda acc, cur: acc + cur, test_data, [])
    for i in range(len(output)):
        not_entity_idx = label2idx['O']
        if output[i, not_entity_idx] >= not_entity_threshold:
            entity = 'O'
        else:
            ent_prob_max = 0
            ent_idx = not_entity_idx
            for j, ent in enumerate(output[i]):
                if ent > ent_prob_max and j != not_entity_idx:
                    ent_prob_max = ent
                    ent_idx = j
            entity = [label for label, idx in label2idx.items() if idx == ent_idx][0]
        classed_output.append((test_data_flat[i][0], entity))
    return classed_output


if __name__ == '__main__':
    main()
