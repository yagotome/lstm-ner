from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense


class NERModel(Sequential):
    def __init__(self, word_embeddings: Embedding, lstm_units: int, num_labels: int, dropout_rate=.5):
        super().__init__()
        self.add(word_embeddings)
        self.add(LSTM(lstm_units, return_sequences=True))
        self.add(Dropout(dropout_rate))
        self.add(LSTM(lstm_units))
        self.add(Dense(num_labels, activation='softmax'))

    def compile(self, **kwargs):
        super.compile(loss='categorical_crossentropy', optimizer='adagrad')

    @staticmethod
    def generate_embedding(vocab_size: int, embedding_dim: int, window_size: int, weights=None):
        embedding = Embedding(vocab_size, embedding_dim, input_length=window_size)
        if weights is not None:
            embedding.set_weights(weights)
            embedding.trainable = True
        return embedding
