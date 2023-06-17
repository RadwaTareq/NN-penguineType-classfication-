#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from random import random
import math


def Activefun(wt_X,activation_fun_name,a=1):
    
    if(activation_fun_name=='Sigmoid'):
        return 1/(1+math.exp(-wt_X))
            
    elif(activation_fun_name=='Hyperbolic'):
        return (1-math.exp(-a*wt_X))/(1+math.exp(-a*wt_X))
    else:
        return 0
    
def calculate_Y(Weight,X,name_activation):
    
    Weight=np.array(Weight)

    wt_X=np.dot(Weight.transpose(),X)
    
    y=Activefun(wt_X,name_activation)
    return y

def Derivative(activation_fun_name, y):
    if (activation_fun_name == 'Sigmoid'):
        return y * (1 - y)
    elif (activation_fun_name == 'Hyperbolic'):
        return (1 - y) * (1 + y)
    else:
        return 0


def calculate_error(Y, outputs):
    np_outputs = np.array(outputs)
    np_outputs.reshape((3, 1))
    np_rowY = np.array(Y)
    np_rowY.reshape((3, 1))

    nameclass = ""

    if (np_outputs.max() == np_outputs[0]):
        nameclass = 0
    elif (np_outputs.max() == np_outputs[1]):
        nameclass = 1
    else:
        nameclass = 2

    nameclass_ = ""
    if (Y[0] == 1 & Y[1] == 0 & Y[2] == 0):
        nameclass_ = 0
    elif (Y[0] == 0 & Y[1] == 1 & Y[2] == 0):
        nameclass_ = 1
    else:
        nameclass_ = 2

    has_same_class = (nameclass_ == nameclass)

    error = np.sum((np_rowY - np_outputs) ** 2)

    return error, nameclass, nameclass_
