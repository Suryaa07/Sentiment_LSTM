import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

reviews = [
    "I loved this movie! It was fantastic.",
    "Terrible acting and a boring plot.",
    "The movie was okay, but not great.",
    "This film is a masterpiece of storytelling."
]
labels = np.array([1, 0, 0, 1])

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)

max_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

x_train, x_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

model = keras.Sequential([
    Embedding(input_dim=10000, output_dim=32, input_length=max_len),
    Bidirectional(LSTM(64)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)

val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=2)
print(f"Validation accuracy: {val_accuracy:.2f}")
