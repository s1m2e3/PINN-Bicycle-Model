import torch
from torch.autograd.functional import jacobian
import numpy as np
import pandas as pd
import datetime

class PINN:

    def __init__(self,functions,n_nodes,input_size,output_size,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh"):
        if len(functions)==output_size:
            raise ValueError("gotta match number of states predicted and diferential equations")
        self.functions = functions

        self.W = torch.randn(size=(n_nodes,input_size))*(high_w-low_w)+low_w
        self.b = torch.randn(size=(n_nodes,1))*(high_b-low_b)+low_b
        
        self.betas = torch.randn(size=(n_nodes,output_size),requires_grad=True)*(high_w-low_w)+low_w
        
    def train(self,accuracy, n_iterations,x_train,y_train,l,rho):

        count = 0
        error = 100
        h = self.get_h(x_train)
        dh = self.get_df(x_train)

        while count < n_iterations or error<accuracy:
            
            error = (y_train-self.predict(x_train))**2/len(y_train)+\
                torch.matmul(dh,self.betas[:,0])-(torch.matmul(dh,self.betas[:,0])**2+torch.matmul(dh,self.betas[:,1])**2)**(1/2)*torch.cos(torch.matmul(h,self.betas[:,2]))+\
                torch.matmul(dh,self.betas[:,1])-(torch.matmul(dh,self.betas[:,0])**2+torch.matmul(dh,self.betas[:,1])**2)**(1/2)*torch.sin(torch.matmul(h,self.betas[:,2]))+\
                torch.matmul(dh,self.betas[:,1])-(torch.matmul(dh,self.betas[:,0])**2+torch.matmul(dh,self.betas[:,1])**2)**(1/2)*torch.sin(torch.matmul(h,self.betas[:,2]))+\
            error.backward()
            self.betas = self.betas + np.multiply(np.linalg.pinv(self.betas.grad),loss) 
            loss = y-self.predict(x_train)
            
            count +=1
            
    def predict(self,x):
        return np.matmul(self.betas,torch.tanh(np.add(np.matmul(x,self.W),self.b)))
    def get_h(self,x):
        return torch.tanh(np.add(np.matmul(x,self.W),self.b))
    def get_dh(self,x):
        return np.multiply((1-self.get_h(x)**2),self.W)
    
        

class PIELM(PINN):
    def __init__(self, functions):
        super().__init__(functions)


class XTFC(PINN):
    def __init__(self, functions):
        super().__init__(functions)

