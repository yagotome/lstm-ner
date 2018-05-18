from keras.utils import np_utils

from lstm_ner import utils
from lstm_ner.ner_model import NERModel


if __name__ == '__main__':
    # defining hyper parameters
    window_size = 3
    dropout_rate = 0.5
    lstm_units = 420
    epochs = 100

    # loading data from files
    embeddings_file = 'data/word_embeddings.txt'
    word_embeddings, word2idx = utils.read_embeddings_file(embeddings_file)
    train_file = 'data/train_data.txt'
    x_train, y_train, label2idx = utils.load_input_output_data(train_file, word2idx, window_size)
    test_file = 'data/test_data.txt'
    x_test, y_test, _ = utils.load_input_output_data(test_file, word2idx, window_size)

    # defining model
    input_length = 2 * window_size + 1
    num_labels = len(label2idx)
    model = NERModel(word_embeddings, lstm_units, num_labels, dropout_rate)

    # summarize the model
    print(model.summary())

    # "binarize" labels
    y_train = np_utils.to_categorical(y_train, num_labels)
    y_test = np_utils.to_categorical(y_test, num_labels)

    # training and eval
    model.fit(x_train, y_train, epochs=epochs)
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Accuracy: %f' % (accuracy * 100))
