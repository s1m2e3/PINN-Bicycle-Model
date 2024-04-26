import torch
from torch.autograd.functional import jacobian
import numpy as np
import pandas as pd
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers):
        super(GRU, self).__init__()
        self.output_size = output_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pre_linear = nn.Linear(output_size,hidden_size).double().to(self.device)
        for param in self.pre_linear.parameters():
            param.requires_grad = False
        self.gru = nn.GRU(input_size,hidden_size,num_layers).double().to(self.device)
        self.post_linear = nn.Linear(hidden_size,output_size).double().to(self.device)
            
    def set_length(self,length):
        length = np.array(length)
        self.length = torch.tensor(length, dtype=torch.double).to(self.device).reshape(1,length.shape[0])
    def set_normalizing_factor(self,normalizing_factor):
        self.normalizing_factor = torch.tensor(normalizing_factor, dtype=torch.double).to(self.device)
    def set_x_min(self,x):
        self.x_min = torch.tensor(x, dtype=torch.double).to(self.device)
    def recover_x(self,x):
        return (x+1)/self.normalizing_factor+self.x_min
    def forward(self,x):
        inputs = x[:,1:3].reshape(x.shape[0],1,2)
        h_0 = torch.zeros(self.gru.num_layers,1,self.gru.hidden_size).double().to(self.device)
        h_0[0,0] = torch.tanh(self.pre_linear(x[0,3:]))
        out = torch.tanh(self.gru(inputs,h_0)[0])
        out = self.post_linear(out)
        return out
    
    def forward_PINN(self,x):
        y = self.forward(x)
        x = self.recover_x(x)
        output_diff_equation = torch.zeros(size = y.shape, dtype=torch.double).to(self.device)
        output_diff_equation[1:,0,0] =y[1:,0,0] - (y[0:-1,0,0]+torch.cos(y[0:-1,0,2])*0.1*x[1:,1])
        output_diff_equation[1:,0,1] =y[1:,0,1] - (y[0:-1,0,1]+torch.sin(y[0:-1,0,2])*0.1*x[1:,1])
        output_diff_equation[1:,0,2] =y[1:,0,2] - (y[0:-1,0,2]+torch.tan(x[1:,2])*0.1*x[1:,1]/self.length[:,1:]/100)
        
        return output_diff_equation
        
class PIELM(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):   
        super(PIELM, self).__init__()
        self.output_size = output_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pre_linear = nn.Linear(input_size,hidden_size).double().to(self.device)
        for param in self.pre_linear.parameters():
            param.requires_grad = False
        self.W = self.pre_linear.weight
        self.b = self.pre_linear.bias.reshape(1,hidden_size)
        self.linear = nn.Linear(hidden_size,output_size).double().to(self.device)
        
    def set_length(self,length):
        length = np.array(length)
        self.length = torch.tensor(length, dtype=torch.double).to(self.device)
    def set_recurrent(self,recurrent):
        self.recurrent = recurrent
    def set_normalizing_factor(self,normalizing_factor):
        self.normalizing_factor = torch.tensor(normalizing_factor, dtype=torch.double).to(self.device)
    def set_x_min(self,x):
        self.x_min = torch.tensor(x, dtype=torch.double).to(self.device)
    def recover_x(self,x):
        return (x+1)/self.normalizing_factor+self.x_min
    def get_dh(self,x):
        return ((1-self.get_h(x)**2)*(self.W[:,0]))
    
    def get_h(self,x):
        return torch.tanh(torch.matmul(x,torch.transpose(self.W,0,1)).add(self.b))
    
    def forward(self,x):
        H = self.get_h(x)
        return self.linear(H)
    def forward_recurrent(self,x):
        x0 = x[0,:]
        y = torch.zeros(size = (x.shape[0],self.output_size), dtype=torch.double).to(self.device)
        for i in range(0,x.shape[0]):
            y0 = self.forward(x0)
            y[i,:] = y0
            if i < x.shape[0]-1:
                x0 = torch.cat((torch.reshape(x[i+1,0:3],(1,3)),y0),1)
        return y
    
    def forward_PINN(self,x):
        HBetas = self.forward(x)
        dH = self.get_dh(x)
        dHBetas = self.linear(dH)*self.normalizing_factor
        x = self.recover_x(x)
        output_diff_equation = torch.zeros(size = dHBetas.shape, dtype=torch.double).to(self.device)
        
        speed = (dHBetas[:,0]**2+dHBetas[:,1]**2)**0.5
        output_diff_equation[:,0] = dHBetas[:,0] - torch.cos(HBetas[:,2]) * speed
        output_diff_equation[:,1] = dHBetas[:,1] - torch.sin(HBetas[:,2]) * speed
        output_diff_equation[:,2] = dHBetas[:,2] - torch.tan(x[:,2]) * speed  / self.length.mean()/100
        return output_diff_equation
