import os

from keras.models import load_model
from keras.utils import np_utils

from lstm_ner import utils
from lstm_ner import ner_model as ner

# defining constants
word_embeddings_file = 'data/word_embeddings.txt'
train_file = 'data/train_data.txt'
test_file = 'data/test_data.txt'
model_file = 'output/model.h5'
char_embeddings_file = 'output/char-embeddings.txt'


if __name__ == '__main__':
    # defining hyper parameters
    word_window_size = 3
    char_window_size = 5
    char_embeddings_dim = 20
    dropout_rate = 0.5
    lstm_units = 420
    conv_num = 10
    epochs = 100

    # loading data from files
    word_embeddings, word2idx, char2idx = utils.read_embeddings_file(word_embeddings_file)
    max_word_len = max(map(lambda word: len(word), word2idx.keys()))
    x_train, y_train, label2idx = utils.load_input_output_data(train_file, word2idx, word_window_size,
                                                               char2idx, max_word_len)
    x_test, y_test, _ = utils.load_input_output_data(test_file, word2idx, word_window_size,
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
        utils.save_embeddings(char_embeddings_file, weights, char2idx)

        # saving whole model
        model.save(model_file)

    # evaluating model
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Accuracy: %f' % (accuracy * 100))
