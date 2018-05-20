from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Merge, Reshape


def generate_model(word_embedding_model, char_embedding_model, lstm_units, num_labels, dropout_rate=.5):
    model = Sequential()
    model.add(Merge([word_embedding_model, char_embedding_model], mode='concat'))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units))
    model.add(Dense(num_labels, activation='softmax'))
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


def generate_char_embedding_model(input_length: int, weights=None, vocab_size=0, embedding_dim=0):
    model = Sequential()
    model.add(generate_embedding(input_length, weights=weights, vocab_size=vocab_size, embedding_dim=embedding_dim))
    return model
