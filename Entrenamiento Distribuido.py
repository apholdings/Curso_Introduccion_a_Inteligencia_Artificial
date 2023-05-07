import time
import numpy as np
import tensorflow as tf

#Pre procesamiento de datos
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#Normalizacion de Imagenes
X_train = X_train / 255.
X_test = X_test / 255.

#Reshaping
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)


# ENTRENAMIENTO DISTRIBUIDO

#definiendo Modelo No-Distribuido
#uusaremos este modelo como comparacion

model_normal = tf.keras.models.Sequential()

model_normal.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))

model_normal.add(tf.keras.layers.Dropout(rate=0.2))

model_normal.add(tf.keras.layers.Dense(units=10, activation='softmax'))

#Compilado
model_normal.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


#Estrategia Entrenamiento Distribuido
distribute = tf.distribute.MirroredStrategy()

with distribute.scope():
    model_distributed = tf.keras.models.Sequential()
    model_distributed.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))
    model_distributed.add(tf.keras.layers.Dropout(rate=0.2))
    model_distributed.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    model_distributed.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

#Diferencia en Tiempo

#Entrenamiento Distribuido
start_time = time.time()
model_distributed.fit(X_train, y_train, epochs=10, batch_size=25)
print("Distributed training took: {}".format(time.time() - start_time))


#Entrenamiento No Distribuido
start_time = time.time()
model_normal.fit(X_train, y_train, epochs=10, batch_size=25)
print("Normal training took: {}".format(time.time() - start_time))

