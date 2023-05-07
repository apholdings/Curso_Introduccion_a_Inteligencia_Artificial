import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as data_reader
#from pandas.util.testing import assert_frame_equal
from collections import deque #Repeticion de Experiencia
from tqdm import tqdm_notebook, tqdm

#IMPLEMENTACION DEL ROBOT
class AI_Trader():
    def __init__(self, state_size, action_space=3, model_name="DataDosis"):
        
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.invetory = []
        self.model_name = model_name
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
        
        self.model = self.model_builder()
        
    def model_builder(self):
        model = tf.keras.models.Sequential()
        
        #Capa Input
        model.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=self.state_size))
        
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        
        #Capa Output
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        
        
        #COMPILADO
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))
        
        return model
        
    def trade(self, state):
        
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        
        actions = self.model.predict(state)
        return np.argmax(actions[0])
        
    def batch_train(self, batch_size):
        
        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])
            
        #Entrenar modelo basado en lote de datos
        for state, action, reward, next_state, done in batch:
            reward = reward
            if not done:
                reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                
            target = self.model.predict(state)
            target[0][action] = reward
            
            #Alimentando Datos a Modelo
            self.model.fit(state, target, epochs=1, verbose=0)
            
        
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
        
#Funcion Sigmoide
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
         
#Price Format
def stocks_price_format(n):
  if n < 0:
    return "- $ {0:2f}".format(abs(n))
  else:
    return "$ {0:2f}".format(abs(n))

#Dataset Loader
def dataset_loader(stock_name):
    
    dataset = data_reader.DataReader(stock_name, data_source="yahoo")
    
    start_date = str(dataset.index[0]).split()[0]
    end_date = str(dataset.index[-1]).split()[0]
    
    close = dataset['Close']
    
    return close
    
def state_creator(data, timestep, window_size):
  
  starting_id = timestep - window_size + 1
  
  if starting_id >= 0:
    windowed_data = data[starting_id:timestep+1]
  else:
    windowed_data = - starting_id * [data[0]] + list(data[0:timestep+1])
    
  state = []
  for i in range(window_size - 1):
    state.append(sigmoid(windowed_data[i+1] - windowed_data[i]))
    
  return np.array([state])

    
#Stock para Tradear
stock_name = "AAPL"
data = dataset_loader(stock_name)


#Definiendo el Modelo y sus Hiper Parametros
window_size = 10
episodes = 1000

batch_size = 32

data_samples = len(data) - 1


#Definir Robot
trader = AI_Trader(window_size)
trader.model.summary()



#LOOP DE ENTRENAMIENTO
for episode in range(1, episodes + 1):
    print("Episode: ()/()".format(episode, episodes))

    state = state_creator(data, 0, window_size + 1)

    total_profit = 0
    trader.inventory = []

    for t in tqdm(range(data_samples)):

        action = trader.trade(state)
        
        next_state = state_creator(data, t+1, window_size + 1)
        reward = 0
        
        if action == 1: #COMPRA
            trader.inventory.append(data[t])
            print("DataDosis Compro: ", stocks_price_format(data[t]))

        elif action == 2 and len(trader.inventory) > 0: #VENTA
            buy_price = trader.inventory.pop(0)
            
            reward = max(data[t] - buy_price, 0)
            total_profit += data[t] - buy_price
            print("DataDosis vendio: ", stocks_price_format(data[t]), " Profit: " + stocks_price_format(data[t] - buy_price) )
            
        if t == data_samples - 1:
            done = True
        else:
            done = False


        trader.memory.append((state, action, reward, next_state, done))
        
        state = next_state

        if done:
            print("########################")
            print("TOTAL PROFIT: {}".format(total_profit))
            print("########################")

        if len(trader.memory) > batch_size:
            trader.batch_train(batch_size)

    if episode % 10 == 0:
        trader.model.save("datadosis_trader_{}.h5".format(episode))
    
    
    
    
    
    
    