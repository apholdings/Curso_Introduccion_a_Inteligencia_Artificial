from tensorflow.keras.datasets import fashion_mnist
import imageio

(X_train, y_train), (X_test, x_train) = fashion_mnist.load_data()

#Guaardando imagenes
for i in range(5):
    imageio.imwrite("uploads/{}.png".format(i), im=X_test[i])
    

#Paso 1: Traer dependencias del proyecto
import os
import requests
import numpy as np
import tensorflow as tf

from imageio import imwrite, imread
from flask import Flask, request, jsonify


#Paso 2: Cargando modelo

#Cargando estructura del modelo
with open("fashion_model_flask.json", "r") as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)

#Traer los pesos
model.load_weights("fashion_model_flask.h5")


#Paso 3: definiendo aplicacion de flask
app = Flask(__name__)

#Funcion de clasificacion para API
@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):
    
    upload_dir = "uploads/"
    
    image = imread(upload_dir + img_name)

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    prediction = model.predict([image.reshape(1, 28*28)])
    
    return jsonify({"object_detected":classes[np.argmax(prediction[0])]})



#Iniciar App de flask y hacer predicciones
app.run(port=5000, debug=False)








































