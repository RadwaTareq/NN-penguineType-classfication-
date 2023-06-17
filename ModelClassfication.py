#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np 
import pandas as pd
from random import random
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# %run Networks.ipynb

#%run YPrediction.ipynb
from YPrediction import * 


# In[7]:


number_classes=3
number_feature=5


class Model:
    def __init__(self,arg):
        self.Data=None
        self.Num_hidden_layer=arg["numHiddenLayer"]
        self.Num_neural_hiddenL=arg["numNeuralHiddenLayer"]
        self.Learning_rate=arg["Learningrate"]
        self.epochs =arg["Epochs"]
        self.bias= arg["Eias"]
        self.Activation_fun=arg["Activation_fun"]
        self.nameData=arg["Data"]
        self.uploadData(arg["Data"])
        self.layers=None

        self.X_training = pd.DataFrame()
        self.X_testing = pd.DataFrame()
        self.Y_traning = pd.DataFrame()
        self.Y_testing = pd.DataFrame()


        self.classes = ['Chinstrap', 'Gentoo', 'Adelie']
        self.features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g']
        
        self.preprocessing()
        self.splitting()
        self.run_model()
        print(arg)
        
        
    def uploadData(self,name):
        if(name=='MNIST'):
            name+='_train'
        name+='.csv'
        name=name.lower()
        self.Data=pd.read_csv(name)

    
    def preprocessing(self):
    
        #fill all missing values 
        for name in self.Data.columns[self.Data.isnull().any()]:
            if (name=='gender'):
                self.Data['gender'].fillna(value='male',inplace=True)
        
            else:
                self.Data[name].fillna(value=self.Data[name].mean(),inplace=True)
                # self.Data.columns[self.Data.isnull().any()]

        species_column = self.Data['species']

        #encoding 
        encoder = LabelEncoder()
        self.Data['gender'] = encoder.fit_transform(self.Data['gender'])

        # one hot encoding
        self.Data = pd.get_dummies(self.Data)

        
        #scaler 
        
        scaler = MinMaxScaler()
        for _feature in self.features:
            self.Data[[_feature]] = scaler.fit_transform(self.Data[[_feature]])
            self.Data[[_feature]] = scaler.fit_transform(self.Data[[_feature]])

        self.Data['species'] = species_column
        self.Data.to_csv('Final.csv', index=False)
        self.Data = pd.read_csv('Final.csv')

    def splitting(self):
        df = self.Data



        for _class in self.classes:
            df1 = df[(df['species'] == _class)]
            x1 = pd.DataFrame()
            for _feature in self.features:
                x1[_feature] = df1[_feature]
            Y1 = df1.iloc[:, 5:8]
            x1_train, x1_test, y1_train, y1_test = train_test_split(x1, Y1, test_size=0.4, random_state=42)
            self.X_training = self.X_training.append(x1_train)
            self.X_testing = self.X_testing.append(x1_test)
            self.Y_traning = self.Y_traning.append(y1_train)
            self.Y_testing = self.Y_testing.append(y1_test)

        value_X0 = 0
        if (self.bias):
            value_X0 = 1

        self.X_training.insert(0, 'bias', np.ones([90, 1]))
        self.X_testing.insert(0, 'bias', np.ones([60, 1]))



    def create_Neuaral(self):
        layers =[]

        for l in range(self.Num_hidden_layer):
            layer=[]
            num_of_weights = 0
            if (self.bias):
                num_of_weights = 1
            if l == 0:
                num_of_weights += number_feature
            else:
                num_of_weights += self.Num_neural_hiddenL[l - 1]
            for neural in range(self.Num_neural_hiddenL[l]):
                neurals=[]
                for n in range(num_of_weights):
                    neurals.append(random())
                layer.append(neurals)    
            layers.append(layer)

        layeroutput=[]

        num_of_weights = 0
        if (self.bias):
            num_of_weights = 1

        num_of_weights += self.Num_neural_hiddenL[self.Num_hidden_layer - 1]

        for neural in range(number_classes):
            neurals=[]
            for n in range(num_of_weights):
                neurals.append(random())
            layeroutput.append(neurals)

        layers.append(layeroutput)

        return layers

    def test(self):
        ConfusionMatrix = np.zeros(shape=(3, 3))

        for [idx, row], [idy, rowY] in zip(self.X_testing.iterrows(), self.Y_testing.iterrows()):
            outputs = self.forward_propagation(row, self.layers)
            error, type_prediction, type_actual = calculate_error(rowY, outputs[-1])
            print(type_actual)
            print(type_prediction)
            print("--------")
            ConfusionMatrix[type_actual][type_prediction] += 1
            ConfusionMatrix[type_actual][type_prediction] += 1
        print(ConfusionMatrix)
        is_true = 0
        for i in range(ConfusionMatrix.shape[0]):
            print(i)
            is_true += int(ConfusionMatrix[i][i])
        is_true = is_true / self.X_testing.shape[0]
        print("Testing accuracy is :" + str(is_true))

    def run_model(self):
        layers = self.create_Neuaral()

        #for i in range(self.Num_hidden_layer):
            #print(self.Num_neural_hiddenL[i])
        # print(layers)
        for ep in range(self.epochs):
            sum = 0

            for [idx, row], [idy, rowY] in zip(self.X_training.iterrows(), self.Y_traning.iterrows()):
                row = np.array(list(row))

                outputs = self.forward_propagation(row, layers)
                for i in range(len(rowY)):
                    if rowY[i] != outputs[-1][i]:
                        gradients = self.backward_propagation(outputs, rowY, layers)
                        new_weights = self.update_weights(layers, gradients, outputs, row)
                        layers=new_weights
                        break
            ConfusionMatrix = np.zeros(shape=(3, 3))
            for [idx, row], [idy, rowY] in zip(self.X_training.iterrows(), self.Y_traning.iterrows()):
                row = np.array(list(row))
                outputs = self.forward_propagation(row, layers)
                error, type_prediction, type_actual = calculate_error(rowY, outputs[-1])
                sum += error
                ConfusionMatrix[type_actual][type_prediction] += 1
            mse=(1/2)*(sum/len(self.X_training))
            print(ConfusionMatrix)
            is_true = 0
            for i in range(ConfusionMatrix.shape[0]):
                print(i)
                is_true += int(ConfusionMatrix[i][i])
            is_true2 = is_true / self.X_training.shape[0]
            print("Training accuracy is :" + str(is_true2))
            self.layers = layers
        self.test()
                # expect=self.Y_traning.values[idx]
                # outputs=np.array(outputs)
                # print(outputs)
                # print(gradients)
                # error=np.sum(np.square(expect-outputs))
                # print(error)

    def update_weights(self, layers, sigma, outputs, row):
        # new_weights = []
        # print(sigma)
        # print(sigma[1])
        for layer_idx in range(self.Num_hidden_layer - 1):
            for neuron_idx in range(self.Num_neural_hiddenL[layer_idx]):
                num_of_weights = 0
                input = []
                if layer_idx == 0:
                    if self.bias:
                        num_of_weights += 1
                        input.append(1)
                    num_of_weights += number_feature
                else:
                    if self.bias:
                        num_of_weights += 1
                        input.append(1)
                    num_of_weights += self.Num_neural_hiddenL[layer_idx - 1]
                for n in range(num_of_weights):
                    if layer_idx == 0:
                        input.extend(row)
                    else:
                        input.extend(outputs[layer_idx - 1])
                    layers[layer_idx][neuron_idx][n] = layers[layer_idx][neuron_idx][n] + self.Learning_rate * sigma[0][layer_idx][neuron_idx] * input[n]
        for neuron_idx in range(number_classes):
            number_of_weights = 0
            input = []
            if self.bias:
                input = [1]
                number_of_weights += 1
            number_of_weights += self.Num_neural_hiddenL[-1]
            input.extend(outputs[-2])
            for n in range(number_of_weights):
                layers[-1][neuron_idx][n] = layers[-1][neuron_idx][n] + self.Learning_rate * sigma[1][neuron_idx] * input[n]
        return layers

    def forward_propagation(self, x_row, layers):
        outputs = []
        for layer_idx in range(self.Num_hidden_layer):
            activation_foreach_layer = []
            if self.bias == False:
                x_row = x_row[1:]
            for neuron_idx in range(self.Num_neural_hiddenL[layer_idx]):
                if layer_idx == 0:
                    activation_foreach_layer.append(calculate_Y(layers[layer_idx][neuron_idx], x_row, self.Activation_fun))
                else:
                    if (self.bias):
                        x_row = [1]
                        x_row.extend(outputs[-1])
                    else:
                        x_row = outputs[-1]
                    activation_foreach_layer.append(calculate_Y(layers[layer_idx][neuron_idx], x_row, self.Activation_fun))
            outputs.append(activation_foreach_layer)

        activation_for_last_layer = []

        if (self.bias):
            x_row = [1]
            x_row.extend(outputs[-1])
        else:
            x_row = outputs[-1]
        for output_neuron_idx in range(number_classes):
            activation_for_last_layer.append(calculate_Y(layers[self.Num_hidden_layer][output_neuron_idx], x_row, self.Activation_fun))
        outputs.append(activation_for_last_layer)

        return outputs

    def backward_propagation(self, outputs, y_actual, layers):

        gradients = []
        activation_output = outputs[-1]
        activation_hidden = outputs[:-1]
        y_act = list(y_actual)
        output_sigma = []
        hidden_sigma = []
        for output_neuron_idx in range(number_classes):
            output_sigma.append((y_act[output_neuron_idx] - activation_output[output_neuron_idx]) * Derivative(self.Activation_fun, activation_output[output_neuron_idx]))

        for layer_idx in reversed(range(self.Num_hidden_layer)):
            hidden_sigma_for_current_layer = []

            for neuron_idx in range(self.Num_neural_hiddenL[layer_idx]):
                sigma_h = 0
                if layer_idx == self.Num_hidden_layer - 1:
                    num_of_weights = 3
                else:
                    num_of_weights = self.Num_neural_hiddenL[layer_idx + 1]
                for x in range(num_of_weights):
                    if layer_idx == self.Num_hidden_layer - 1:
                        next_sigma = output_sigma[x]
                    else:

                        next_sigma = hidden_sigma[0][x]
                    sigma_h += (Derivative(self.Activation_fun, activation_hidden[layer_idx][neuron_idx]) * next_sigma * layers[layer_idx + 1][x][neuron_idx])
                hidden_sigma_for_current_layer.append(sigma_h)
            hidden_sigma.insert(0, hidden_sigma_for_current_layer)



        gradients.append(hidden_sigma)
        gradients.append(output_sigma)
        return gradients




            

