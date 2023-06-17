#!/usr/bin/env python
# coding: utf-8

# In[6]:


from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox

#%run ModelClassfication.ipynb
from ModelClassfication import *


# In[7]:




class GUI :
    
    def __init__(self,tk_parent):
        self.tk=tk_parent
        self.Num_hidden_layer=None
        self.Num_neural_hiddenL=[]
        self.Learning_rate=None
        self.epochs =None
        self.bias= None
        self.Activation_fun=None
        self.Data=None
        
        self.button=Button(self.tk ,text="Classification",width=20,command=self.runModel)
        self.button.place(x = 270, y = 570)
        
    def runModel(self):
        
        input_Model={}
        input_Model["Data"]=self.Data.get()
        input_Model["numHiddenLayer"]=int(self.Num_hidden_layer.get())
        input_Model["numNeuralHiddenLayer"]=[int(i) for i in self.Num_neural_hiddenL.get().split(',')]
        input_Model["Learningrate"]=float(self.Learning_rate.get())
        input_Model["Epochs"]=int(self.epochs.get())
        input_Model["Activation_fun"]=self.Activation_fun.get()
        input_Model["Eias"]=self.bias.get()=="YES"
        
        Model(input_Model)
        
        
        
     
    def runComponents(self):
        self.hiddenlayer()
        self.ActivationFunction()
        self.Bias()
        self.Epochs()
        self.LearningRate()
        self.neuralHiddenLayer()
        self.chooseData()
        
    def chooseData(self):
        choose =["MNIST" , "Penguins"]
        chooseData_Label=Label(self.tk,text = "Data", font=("Arial", 12)).place(x = 70, y = 60)
        self.Data=Combobox(self.tk,width = 15,values=choose,textvariable = StringVar())
        self.Data.place(x = 300, y = 60)
        
    
    def hiddenlayer(self):
        hiddenlayerLabel=Label(self.tk,text = "Enter # hidden Layer", font=("Arial", 12)).place(x = 70, y = 120)
        self.Num_hidden_layer=Entry(self.tk)
        self.Num_hidden_layer.place(x = 300, y = 120)
        
        
    def neuralHiddenLayer(self):
        hiddenlayerLabel=Label(self.tk,text = "Enter # neural hidden Layer", font=("Arial", 12)).place(x = 70, y = 190)
        self.Num_neural_hiddenL=Entry(self.tk)
        self.Num_neural_hiddenL.place(x = 300, y = 190)

        
    def LearningRate(self):
        Learning_rateLabel=Label(self.tk,text = "Enter  Learning_rate", font=("Arial", 12)).place(x = 70, y = 260)
        self.Learning_rate=Entry(self.tk)
        self.Learning_rate.place(x = 300, y = 260)
        
        
    def Epochs(self):
        Learning_rateLabel=Label(self.tk,text = "Enter  Epochs", font=("Arial", 12)).place(x = 70, y = 360)
        self.epochs=Entry(self.tk)
        self.epochs.place(x = 300, y = 360)
        
    def Bias(self):
        choose=["NO","YES"]
        bias_Label=Label(self.tk,text = "Enter Bias", font=("Arial", 12)).place(x = 70, y = 420)
        self.bias=Combobox(self.tk,width = 15,values=choose,textvariable = StringVar())
        self.bias.place(x = 300, y = 420)
        
    def ActivationFunction(self):
        choose =["Sigmoid" , "Hyperbolic"]
        activation_function_Label=Label(self.tk,text = "Activation Function", font=("Arial", 12)).place(x = 70, y = 500)
        self.Activation_fun=Combobox(self.tk,width = 15,values=choose,textvariable = StringVar())
        self.Activation_fun.place(x = 300, y = 500)
        
        
        
        
        
       
        
    

        
        
        
def main():#run loop main
    root = Tk()
    root.geometry("500x900")
    d=GUI(root)
    d.runComponents()
    root.mainloop()
    
main()    


# In[ ]:





# In[ ]:





# In[ ]:




