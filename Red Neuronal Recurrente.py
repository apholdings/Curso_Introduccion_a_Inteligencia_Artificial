import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb


#Parametros
number_of_words = 20000
max_len = 100

#Cargando Set de Datos
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = number_of_words)

#Padding
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen = max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen = max_len)

#Embedding Parameters
vocab_size = number_of_words

embed_size = 128


#CREANDO RED NEURONAL RECURRENTE
model = tf.keras.Sequential()

# Capa Embed
model.add(tf.keras.layers.Embedding(vocab_size, embed_size, input_shape=(X_train.shape[1],)))

# Capa LSTM
model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))

# Capa Dense
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#ENTRENANDO RED
# Compilar
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento
model.fit(X_train, y_train, epochs=3, batch_size=128)


#EVALUADO
test_loss, test_accuracy = model.evaluate(X_test, y_test)














