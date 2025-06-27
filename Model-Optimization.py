# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:56:02 2021

@author: kumar
"""

import pandas as pd
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def type_to_numeric(x):
    if x=='setosa':
        return 0
    if x=='versicolor':
        return 1
    else :
        return 2

def get_data():
    
    os.chdir("D:\Dropbox\Dropbox\Learning\Deep Learning\Tensorflow-Learning")
    
    iris_data = pd.read_csv("iris.csv")
    
    iris_data.dtypes
    iris_data.describe()
    iris_data.head()
    
    #Use a Label encoder to convert String to numeric values for the target variable

    label_encoder = preprocessing.LabelEncoder()
    iris_data['Species'] = label_encoder.fit_transform(
                                    iris_data['Species'])
    
    #Convert input to numpy array
    np_iris = iris_data.to_numpy()
    
    #Separate feature and target variables
    X_data = np_iris[:,0:4]
    Y_data=np_iris[:,4]
    
    print("\nFeatures before scaling :\n------------------------------------")
    print(X_data[:5,:])
    print("\nTarget before scaling :\n------------------------------------")
    print(Y_data[:5])
    
    #Create a scaler model that is fit on the input data.
    scaler = StandardScaler().fit(X_data)
    
    #Scale the numeric feature variables
    X_data = scaler.transform(X_data)
    
    #Convert target variable as a one-hot-encoding array
    Y_data = tf.keras.utils.to_categorical(Y_data,3)


    return X_data,Y_data

def base_model_config():
    model_config = {
            "HIDDEN_NODES" : [32,64],
            "HIDDEN_ACTIVATION" : "relu",
            "OUTPUT_NODES" : 3,
            "OUTPUT_ACTIVATION" : "softmax",
            "WEIGHTS_INITIALIZER" : "random_normal",
            "BIAS_INITIALIZER" : "zeros",
            "NORMALIZATION" : "none",
            "OPTIMIZER" : "rmsprop",
            "LEARNING_RATE" : 0.001,
            "REGULARIZER" : None,
            "DROPOUT_RATE" : 0.0,
            "EPOCHS" : 20,
            "BATCH_SIZE" : 16,
            "VALIDATION_SPLIT" : 0.1,
            "VERBOSE" : 0,
            "LOSS_FUNCTION" : "categorical_crossentropy",
            "METRICS" : ["accuracy"]
            }
    return model_config

def get_optimizer(optimizer_name, learning_rate):
    #'sgd','rmsprop','adam','adagrad'
    optimizer=None
    
    if optimizer_name == 'adagrad': 
        optimizer = keras.optimizers.Adagrad(lr=learning_rate)

    elif 'rmsprop':
        optimizer = keras.optimizers.RMSprop(lr=learning_rate)

    elif'adam' :
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        
    else :
        optimizer = keras.optimizers.SGD(lr=learning_rate)
            
    return optimizer
    
    
def create_and_run_model(model_config,X,Y) :
    
    model=tf.keras.models.Sequential()
    
    for layer in range(len(model_config["HIDDEN_NODES"])):
        
        if (layer == 0):
            model.add(
                    keras.layers.Dense(model_config["HIDDEN_NODES"][layer],
                    input_shape=(X.shape[1],),
                    name="Dense-Layer-" + str(layer),
                    kernel_initializer = model_config["WEIGHTS_INITIALIZER"],
                    bias_initializer = model_config["BIAS_INITIALIZER"],
                    kernel_regularizer=model_config["REGULARIZER"],
                    activation=model_config["HIDDEN_ACTIVATION"]))
        else:
            
            if ( model_config["NORMALIZATION"] == "batch"):
                model.add(keras.layers.BatchNormalization())
                
            if ( model_config["DROPOUT_RATE"] > 0.0 ):
                model.add(keras.layers.Dropout(model_config["DROPOUT_RATE"]))
                
            model.add(
                    keras.layers.Dense(model_config["HIDDEN_NODES"][layer],
                    name="Dense-Layer-" + str(layer),
                    kernel_initializer = model_config["WEIGHTS_INITIALIZER"],
                    bias_initializer = model_config["BIAS_INITIALIZER"],
                    kernel_regularizer=model_config["REGULARIZER"],
                    activation=model_config["HIDDEN_ACTIVATION"])) 
            

            
    model.add(keras.layers.Dense(model_config["OUTPUT_NODES"],
                    name="Output-Layer",
                    activation=model_config["OUTPUT_ACTIVATION"]))
    
    optimizer = get_optimizer( model_config["OPTIMIZER"],
                              model_config["LEARNING_RATE"])
    
    model.compile(loss=model_config["LOSS_FUNCTION"],
                  optimizer=optimizer,
                   metrics=model_config["METRICS"])
    
    model.summary()
    
    history=model.fit(X,
          Y,
          batch_size=model_config["BATCH_SIZE"],
          epochs=model_config["EPOCHS"],
          verbose=model_config["VERBOSE"],
          validation_split=model_config["VALIDATION_SPLIT"])
    
    return history

def plot_graph(accuracy_measures, title):
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(20,20))
    for experiment in accuracy_measures.keys():
        plt.plot(accuracy_measures[experiment], label=experiment)
        
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    

#################################################################
## 1. Number of layers
#################################################################
model_config = base_model_config()
X,Y = get_data()
accuracy_measures = {}

layer_list =[]
for layer_count in range(5):
    
    layer_list.append(32)
    
    model_config = base_model_config()
    X,Y = get_data()
    
    model_config["HIDDEN_NODES"] = layer_list
    history=create_and_run_model(model_config,X,Y)
    
    accuracy_measures["layers :" + str(layer_count)] = history.history["accuracy"]

plot_graph(accuracy_measures, "Compare Hidden Layers")

#################################################################
## 2.Number of nodes in a layer
#################################################################

model_config = base_model_config()
X,Y = get_data()
accuracy_measures = {}

node_increment=8

for node_count in range(1,5):
    
    #have 3 hidden layers in the networks
    layer_list =[]
    for layer_count in range(3):
        layer_list.append(node_count * node_increment)
    
    model_config = base_model_config()
    X,Y = get_data()
    
    model_config["HIDDEN_NODES"] = layer_list
    history=create_and_run_model(model_config,X,Y)
    
    accuracy_measures["nodes : " + str(node_count * node_increment)] = history.history["accuracy"]


plot_graph(accuracy_measures, "Compare Nodes in a Layer")

#################################################################
## 3. Activation Functions
#################################################################

model_config = base_model_config()
X,Y = get_data()
accuracy_measures = {}

activation_list = ['relu','sigmoid','tanh']
for activation in activation_list:
    
    model_config = base_model_config()
    X,Y = get_data()
    
    model_config["HIDDEN_ACTIVATION"] = activation
    history=create_and_run_model(model_config,X,Y)
    
    accuracy_measures["Activation : " + activation] = history.history["accuracy"]

plot_graph(accuracy_measures, "Compare Activiation Functions")

#################################################################
## 4. Normalization 
#################################################################

model_config = base_model_config()
X,Y = get_data()
accuracy_measures = {}

normalization_list = ['none','batch']
for normalization in normalization_list:
    
    model_config = base_model_config()
    X,Y = get_data()
    
    model_config["NORMALIZATION"] = normalization
    history=create_and_run_model(model_config,X,Y)
    
    accuracy_measures["Normalization : " + normalization] = history.history["accuracy"]

plot_graph(accuracy_measures, "Compare Normalization Techniques")


#################################################################
## 5. Optimizers 
#################################################################

model_config = base_model_config()
X,Y = get_data()
accuracy_measures = {}

optimizer_list = ['sgd','rmsprop','adam','adagrad']
for optimizer in optimizer_list:
    
    model_config = base_model_config()
    X,Y = get_data()
    
    model_config["OPTIMIZER"] = optimizer
    history=create_and_run_model(model_config,X,Y)
    
    accuracy_measures["Optimizer : " + optimizer] = history.history["accuracy"]


plot_graph(accuracy_measures, "Compare Optimizers")

#################################################################
## 6. Regularization 
#################################################################

model_config = base_model_config()
X,Y = get_data()
accuracy_measures = {}

regularizer_list = ['l1','l2','l1_l2']
for regularizer in regularizer_list:
    
    model_config = base_model_config()
    X,Y = get_data()
    
    model_config["REGULARIZER"] = regularizer
    history=create_and_run_model(model_config,X,Y)
    
    accuracy_measures["Regularizer : " + regularizer] = history.history["accuracy"]

plot_graph(accuracy_measures, "Compare Regularizers")

#################################################################
## 7. Dropout 
#################################################################

model_config = base_model_config()
X,Y = get_data()
accuracy_measures = {}

dropout_list = [0.0, 0.1, 0.2, 0.3]
for dropout in dropout_list:
    
    model_config = base_model_config()
    X,Y = get_data()
    
    model_config["DROPOUT_RATE"] = dropout
    history=create_and_run_model(model_config,X,Y)
    
    #Using validation accuracy
    accuracy_measures["Dropout : " + str(dropout)] = history.history["val_accuracy"]

plot_graph(accuracy_measures, "Compare Dropout Rates")

#################################################################
## 8. Learning Rate (goes with optimizer)
#################################################################

model_config = base_model_config()
X,Y = get_data()
accuracy_measures = {}

learning_rate_list = [0.001, 0.005,0.01,0.1]
for learning_rate in learning_rate_list:
    
    model_config = base_model_config()
    X,Y = get_data()
    
    model_config["LEARNING_RATE"] = learning_rate
    history=create_and_run_model(model_config,X,Y)
    
    #Using validation accuracy
    accuracy_measures["Learning Rate : " + str(learning_rate)] = history.history["accuracy"]

plot_graph(accuracy_measures, "Compare Learning Rates")
    
#################################################################
## 9. Weights Initialization 
#################################################################

model_config = base_model_config()
X,Y = get_data()
accuracy_measures = {}

initializer_list = ['random_normal','zeros','ones',"random_uniform"]
for initializer in initializer_list:
    
    model_config = base_model_config()
    X,Y = get_data()
    
    model_config["WEIGHTS_INITIALIZER"] = initializer
    history=create_and_run_model(model_config,X,Y)
    
    accuracy_measures["Initializer : " + initializer] = history.history["accuracy"]


plot_graph(accuracy_measures, "Compare Weights Initializers")