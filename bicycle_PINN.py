import torch
from torch.autograd.functional import jacobian
import numpy as np
import pandas as pd
import datetime

class PIELM:

    def __init__(self,n_nodes,input_size,output_size,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh"):
        # if len(functions)==output_size:
        #     raise ValueError("gotta match number of states predicted and diferential equations")
        # self.functions = functions

        self.W = (torch.randn(size=(n_nodes,1),dtype=torch.double)*(high_w-low_w)+low_w)
        self.b = (torch.randn(size=(n_nodes,1),dtype=torch.double)*(high_b-low_b)+low_b)
        
        self.betas = (torch.randn(size=(n_nodes,output_size),requires_grad=True,dtype=torch.double)*(high_w-low_w)+low_w)
        
    def train(self,accuracy, n_iterations,x_train,y_train,l,rho):

        count = 0
        error = 100
        x_train = torch.tensor(x_train.values).reshape(x_train.shape[0],1)
        y_train = torch.tensor(y_train.values)
        l = torch.tensor(l.values)
        rho = torch.tensor(rho.values)
        h = self.get_h(x_train)
        dh = self.get_dh(x_train)
        self.betas = torch.matmul(torch.linalg.pinv(h),y_train)
        self.betas.requires_grad=True
        while count < n_iterations or error<accuracy:
            self.betas.retain_grad()
            
            l_pred = (y_train-self.predict(x_train))
            l_x = torch.matmul(dh,self.betas[:,0])-(torch.matmul(dh,self.betas[:,0])**2+torch.matmul(dh,self.betas[:,1])**2)**(1/2)*torch.cos(torch.matmul(h,self.betas[:,2]))
            l_y = torch.matmul(dh,self.betas[:,1])-(torch.matmul(dh,self.betas[:,0])**2+torch.matmul(dh,self.betas[:,1])**2)**(1/2)*torch.sin(torch.matmul(h,self.betas[:,2])) 
            l_theta = torch.matmul(dh,self.betas[:,2])-(torch.matmul(dh,self.betas[:,0])**2+torch.matmul(dh,self.betas[:,1])**2)**(1/2)*torch.tan(torch.matmul(h,self.betas[:,3]))/l
            l_delta = torch.matmul(dh,self.betas[:,3])-rho
            loss_dh = torch.stack((l_x,l_y,l_theta,l_delta),dim=1)
            
            loss = torch.sum((torch.cat((l_pred,loss_dh),dim=0))**2)/len(y_train)
            loss.backward(retain_graph=True)
            self.betas = self.betas + torch.transpose(torch.mul(loss,torch.linalg.pinv(self.betas.grad)),0,1)
            count +=1
            
    def predict(self,x):
        return torch.matmul(self.get_h(x),self.betas)
    def get_h(self,x):
        return torch.tanh(torch.add(torch.matmul(x,torch.transpose(self.W,0,1)),torch.transpose(self.b,0,1)))
    def get_dh(self,x):
        return torch.mul((1-self.get_h(x)**2),torch.transpose(self.W,0,1))
    

# class XTFC(PIELM):
#     def __init__(self,functions,n_nodes,input_size,output_size,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh"):
#         super().__init__(functions,n_nodes,input_size,output_size,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh")

#     def train(self,accuracy, n_iterations,x_train,y_train,l,rho):

#         count = 0
#         error = 100
#         h = self.get_h(x_train)
#         dh = self.get_df(x_train)

#         while count < n_iterations or error<accuracy:
            
#             l_pred = (y_train-self.predict(x_train))
#             l_x = torch.matmul(dh,self.betas[:,0])-(torch.matmul(dh,self.betas[:,0])**2+torch.matmul(dh,self.betas[:,1])**2)**(1/2)*torch.cos(torch.matmul(h,self.betas[:,2]))
#             l_y = torch.matmul(dh,self.betas[:,1])-(torch.matmul(dh,self.betas[:,0])**2+torch.matmul(dh,self.betas[:,1])**2)**(1/2)*torch.sin(torch.matmul(h,self.betas[:,2])) 
#             l_theta = torch.matmul(dh,self.betas[:,2])-(torch.matmul(dh,self.betas[:,0])**2+torch.matmul(dh,self.betas[:,1])**2)**(1/2)*torch.tan(torch.matmul(h,self.betas[:,3]))/l
#             l_delta = torch.matmul(dh,self.betas[:,3])-rho
#             loss = torch.cat(l_pred,l_x,l_y,l_theta,l_delta)
#             loss.backward()
#             self.betas = self.betas + np.multiply(np.linalg.pinv(self.betas.grad),loss) 
#             error = loss**2/len(y_train)
#             count +=1
