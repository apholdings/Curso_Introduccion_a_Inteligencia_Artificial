import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import fashion_mnist

#Pre procesamiento de datos
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#Normalizacion de las imagenes
X_train = X_train / 255.
X_test = X_test / 255.

#Reshaping
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)


#Definicion de Modelo
model = tf.keras.models.Sequential()

#Armando Capas

#Primera Capa Capa Input
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))

#Segunda Capa
model.add(tf.keras.layers.Dropout(rate=0.2))

#Capa de Salida
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

#Compilado
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

#Entrenando Modelo
model.fit(X_train, y_train, epochs=5)

#Guardando Pesos del Modelo
model_name='fashion_mobile_model.h5'
tf.keras.models.save_model(model, model_name)

#Creando Conversion a TF Lite
converter = tf.lite.TFLiteConverter.from_keras_model_file(model_name)

#Convirtiendo modelo
tflite_model = converter.convert()

#Guarrdando Archivo TFLite
with open("tf_model.tflite", "wb") as f:
  f.write(tflite_model)


















