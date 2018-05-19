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
        super().compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

    @staticmethod
    def generate_embedding(input_length: int, weights=None, vocab_size=0, embedding_dim=0):
        if weights is not None:
            vocab_size = weights.shape[0]
            embedding_dim = weights.shape[1]
            return Embedding(vocab_size, embedding_dim, input_length=input_length, weights=[weights], trainable=False)
        return Embedding(vocab_size, embedding_dim, input_length=input_length)
