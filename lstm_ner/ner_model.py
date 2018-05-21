from keras import Sequential, Model
from keras.layers import Embedding, LSTM, Dropout, Dense, Reshape, Conv1D, MaxPooling1D, TimeDistributed, \
    concatenate


def generate_model(word_embedding_model: Sequential, char_embedding_model: Sequential, lstm_units: int, num_labels: int,
                   dropout_rate=.5):
    merged_input_models = concatenate([word_embedding_model.output, char_embedding_model.output])
    core_model = Sequential()
    core_model.add(LSTM(lstm_units,
                        input_shape=(None, word_embedding_model.output_shape[2] + char_embedding_model.output_shape[2]),
                        return_sequences=True))
    core_model.add(Dropout(dropout_rate))
    core_model.add(LSTM(lstm_units))
    core_model.add(Dense(num_labels, activation='softmax'))
    core_model_output = core_model(merged_input_models)
    model = Model([word_embedding_model.input, char_embedding_model.input], core_model_output)
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    return model


def generate_embedding(input_length: int, weights=None, vocab_size=0, embedding_dim=0):
    if weights is not None:
        vocab_size = weights.shape[0]
        embedding_dim = weights.shape[1]
        return Embedding(vocab_size, embedding_dim, input_length=input_length, weights=[weights], trainable=False)
    return Embedding(vocab_size, embedding_dim, input_length=input_length)


def generate_word_embedding_model(input_length: int, weights=None, vocab_size=0, embedding_dim=0):
    model = Sequential()
    model.add(generate_embedding(input_length, weights=weights, vocab_size=vocab_size, embedding_dim=embedding_dim))
    return model


def generate_char_embedding_model(max_word_len: int, max_word_len_padded: int, word_input_len: int,
                                  char_embedding_dim: int, conv_num: int, char_window_size,
                                  vocab_size: int):
    char_input_len = word_input_len * max_word_len_padded
    model = Sequential()
    model.add(generate_embedding(char_input_len, vocab_size=vocab_size, embedding_dim=char_embedding_dim))
    model.add(Reshape((word_input_len, max_word_len_padded, char_embedding_dim)))
    model.add(TimeDistributed(Conv1D(conv_num, char_window_size)))
    model.add(TimeDistributed(MaxPooling1D(max_word_len)))
    model.add(Reshape((word_input_len, conv_num)))
    return model
