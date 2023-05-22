import torch
from torch.autograd.functional import jacobian
import numpy as np
import pandas as pd
import datetime
import torch.nn as nn

class PIELM:

    def __init__(self,n_nodes,input_size,output_size,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh"):
        # if len(functions)==output_size:
        #     raise ValueError("gotta match number of states predicted and diferential equations")
        # self.functions = functions
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nodes = n_nodes
        self.W = (torch.randn(size=(n_nodes,1),dtype=torch.float)*(high_w-low_w)+low_w)
        self.b = (torch.randn(size=(n_nodes,1),dtype=torch.float)*(high_b-low_b)+low_b)
        
        self.betas = torch.zeros(size=(output_size*n_nodes,),requires_grad=True,dtype=torch.float)+1
        

    def train(self,accuracy, n_iterations,x_train,y_train,l,rho):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        count = 0
        error = 100
        
        self.x_train = torch.tensor(x_train,dtype=torch.float).reshape(x_train.shape[0],1)
        self.y_train = torch.tensor(y_train,dtype=torch.float)
        self.x_train_pred = self.x_train[0:len(self.y_train),]

        self.l = torch.tensor(l,dtype=torch.float)
        
        self.rho = torch.tensor(rho,dtype=torch.float)
        
        h = self.get_h(self.x_train)
        #self.W.to(device)
        #self.b.to(device)
        
        #self.betas.to(device)
        print(self.betas.is_cuda)
        print("number of samples:",len(self.x_train))
        while count < n_iterations:
            
            with torch.no_grad():
                jac = jacobian(self.predict_jacobian,self.betas)
                loss = self.predict_loss(self.x_train,self.y_train,self.x_train_pred)
                pinv_jac = torch.linalg.pinv(jac)
                
                delta = torch.matmul(pinv_jac,loss)
                self.betas -=delta*0.1
            if count %10==0:
                print(loss.abs().max(dim=0),loss.mean(dim=0))
                #print(torch.mean(loss[0:4]))
                #print(torch.max(loss[0:4]))
                #print(torch.min(loss[0:4]))
                print("final loss:",(loss**2).mean())
            count +=1
        print(loss[0:20],"x position")
        print(loss[20:40],"y position")
        print(loss[40:60],"angle")
        print(loss[60:80],"steering")        
        print(loss[80:100],"x speed")
        print(loss[100:120],"y speed")
        print(loss[120:140],"angular speed")
        print(loss[140:160],"delta rate")        
        
        
        # for epoch in range(n_iterations):
        
            
        #     self.optimizer.zero_grad()
        #     outputs = self.forward(x_train_data)
        #     loss = self.criterion(outputs, y_train_data)
        #     loss.backward()
        #     self.optimizer.step()
            
        #     #Print training statistics
        #     if (epoch+1) % 10 == 0:
        #         print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
        #             .format(epoch+1, num_epochs, epoch+1, len(x_train_data), loss.item()))

    

    
    def predict_jacobian(self,betas):
        
       
        l_pred_x = self.y_train[:,0]-torch.matmul(self.get_h(self.x_train_pred),betas[0:self.nodes])
        l_pred_y = self.y_train[:,1]-torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes:2*self.nodes])
        l_pred_theta = self.y_train[:,2]-torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*2:3*self.nodes])
        l_pred_delta = self.y_train[:,3]-torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*3:4*self.nodes])
        
        l_x = torch.matmul(self.get_dh(self.x_train),betas[0:self.nodes])-(torch.matmul(self.get_dh(self.x_train),betas[0:self.nodes])**2+torch.matmul(self.get_dh(self.x_train),betas[self.nodes:2*self.nodes])**2)**(1/2)*torch.cos(torch.matmul(self.get_h(self.x_train),betas[self.nodes*2:3*self.nodes]))
        l_y = torch.matmul(self.get_dh(self.x_train),betas[self.nodes:2*self.nodes])-(torch.matmul(self.get_dh(self.x_train),betas[0:self.nodes])**2+torch.matmul(self.get_dh(self.x_train),betas[self.nodes:2*self.nodes])**2)**(1/2)*torch.sin(torch.matmul(self.get_h(self.x_train),betas[self.nodes*2:3*self.nodes])) 
        l_theta = torch.matmul(self.get_dh(self.x_train),betas[self.nodes*2:3*self.nodes])-(torch.matmul(self.get_dh(self.x_train),betas[0:self.nodes])**2+torch.matmul(self.get_dh(self.x_train),betas[self.nodes:2*self.nodes])**2)**(1/2)*torch.tan(torch.matmul(self.get_h(self.x_train),betas[self.nodes*2:3*self.nodes]))/self.l
        l_delta = torch.matmul(self.get_dh(self.x_train),betas[self.nodes*3:4*self.nodes])-self.rho
        
        loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        
        return loss
            
    def predict_loss(self,x,y,x_pred):
       
        l_pred_x = y[:,0]-torch.matmul(self.get_h(x_pred),self.betas[0:self.nodes])
        l_pred_y = y[:,1]-torch.matmul(self.get_h(x_pred),self.betas[self.nodes:2*self.nodes])
        l_pred_theta = y[:,2]-torch.matmul(self.get_h(x_pred),self.betas[self.nodes*2:3*self.nodes])
        l_pred_delta = y[:,3]-torch.matmul(self.get_h(x_pred),self.betas[self.nodes*3:4*self.nodes])
        
        l_x = torch.matmul(self.get_dh(x),self.betas[0:self.nodes])-(torch.matmul(self.get_dh(x),self.betas[0:self.nodes])**2+torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])**2)**(1/2)*torch.cos(torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes]))
        l_y = torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])-(torch.matmul(self.get_dh(x),self.betas[0:self.nodes])**2+torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])**2)**(1/2)*torch.sin(torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes])) 
        l_theta = torch.matmul(self.get_dh(x),self.betas[self.nodes*2:3*self.nodes])-(torch.matmul(self.get_dh(x),self.betas[0:self.nodes])**2+torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])**2)**(1/2)*torch.tan(torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes]))/self.l
        l_delta = torch.matmul(self.get_dh(x),self.betas[self.nodes*3:4*self.nodes])-self.rho
        
        loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        
        return loss    
    
    def get_h(self,x):
        return torch.tanh(torch.add(torch.matmul(x,torch.transpose(self.W,0,1)),torch.transpose(self.b,0,1)))
    def get_dh(self,x):
        return torch.mul((1-self.get_h(x)**2),torch.transpose(self.W,0,1))
    
    def pred(self,x):
        
        x = torch.tensor(np.array(x),dtype=torch.float).reshape(x.shape[0],1)
        
        x_pred = torch.matmul(self.get_h(x),self.betas[0:self.nodes]) 
        y_pred = torch.matmul(self.get_h(x),self.betas[self.nodes:2*self.nodes])
        theta_pred = torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes])
        delta_pred = torch.matmul(self.get_h(x),self.betas[self.nodes*3:4*self.nodes])
        return torch.vstack((x_pred,y_pred,theta_pred,delta_pred))

class XTFC(PIELM):
    def __init__(self,n_nodes,input_size,output_size,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh"):
        super().__init__(n_nodes,input_size,output_size,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh")

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

    def predict_jacobian(self,betas):
        
        x0 = -torch.matmul(self.get_h(self.x_train_pred[0]),betas[0:self.nodes])[0]+self.y_train[0,0]
        print(x0)
        y0 = -torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes:2*self.nodes])[0]+self.y_train[0,1]
        theta0 = -torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*2:3*self.nodes])[0]+self.y_train[0,2]
        delta0 =-torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*3:4*self.nodes])[0]+self.y_train[0,3]

        hx = torch.matmul(self.get_h(self.x_train_pred),betas[0:self.nodes]) + x0
        hy = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes:2*self.nodes]) +y0
        htheta = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*2:3*self.nodes]) +theta0
        hdelta = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*3:4*self.nodes]) +delta0

        l_pred_x = self.y_train[:,0]-hx
        l_pred_y = self.y_train[:,1]-hy
        l_pred_theta = self.y_train[:,2]-htheta
        l_pred_delta = self.y_train[:,3]-hdelta
        
        l_x = torch.matmul(self.get_dh(self.x_train),betas[0:self.nodes])-\
        (torch.matmul(self.get_dh(self.x_train),betas[0:self.nodes])**2+\
        torch.matmul(self.get_dh(self.x_train),betas[self.nodes:2*self.nodes])**2)**(1/2)\
        *torch.cos(torch.matmul(self.get_h(self.x_train),betas[self.nodes*2:3*self.nodes])+theta0)
        
        l_y = torch.matmul(self.get_dh(self.x_train),betas[self.nodes:2*self.nodes])-\
        (torch.matmul(self.get_dh(self.x_train),betas[0:self.nodes])**2+\
        torch.matmul(self.get_dh(self.x_train),betas[self.nodes:2*self.nodes])**2)**(1/2)\
        *torch.sin(torch.matmul(self.get_h(self.x_train),betas[self.nodes*2:3*self.nodes])+theta0) 
        
        l_theta = torch.matmul(self.get_dh(self.x_train),betas[self.nodes*2:3*self.nodes])-\
        (torch.matmul(self.get_dh(self.x_train),betas[0:self.nodes])**2\
        +torch.matmul(self.get_dh(self.x_train),betas[self.nodes:2*self.nodes])**2)**(1/2)\
        *torch.tan(torch.matmul(self.get_h(self.x_train),betas[self.nodes*2:3*self.nodes])+delta0)/self.l
        
        l_delta = torch.matmul(self.get_dh(self.x_train),betas[self.nodes*3:4*self.nodes])-self.rho
        
        loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        
        return loss
            
    def predict_loss(self,x,y,x_pred):

        x0 = -torch.matmul(self.get_h(x_pred),self.betas[0:self.nodes])[0]+y[0,0]
        y0 = -torch.matmul(self.get_h(x_pred),self.betas[self.nodes:2*self.nodes])[0]+y[0,1]
        theta0 = -torch.matmul(self.get_h(x_pred),self.betas[self.nodes*2:3*self.nodes])[0]+y[0,2]
        delta0 =-torch.matmul(self.get_h(x_pred),self.betas[self.nodes*3:4*self.nodes])[0]+y[0,3]

        hx = torch.matmul(self.get_h(x_pred),self.betas[0:self.nodes]) + x0
        hy = torch.matmul(self.get_h(x_pred),self.betas[self.nodes:2*self.nodes]) +y0
        htheta = torch.matmul(self.get_h(x_pred),self.betas[self.nodes*2:3*self.nodes]) +theta0
        hdelta = torch.matmul(self.get_h(x_pred),self.betas[self.nodes*3:4*self.nodes]) +delta0

        l_pred_x = y[:,0]-hx
        l_pred_y = y[:,1]-hy
        l_pred_theta = y[:,2]-htheta
        l_pred_delta = y[:,3]-hdelta
        
        l_x = torch.matmul(self.get_dh(x),self.betas[0:self.nodes])-\
        (torch.matmul(self.get_dh(x),self.betas[0:self.nodes])**2+\
        torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])**2)**(1/2)\
        *torch.cos(torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes])+theta0)
        
        l_y = torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])-\
        (torch.matmul(self.get_dh(x),self.betas[0:self.nodes])**2+\
        torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])**2)**(1/2)\
        *torch.sin(torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes])+theta0) 
        
        l_theta = torch.matmul(self.get_dh(x),self.betas[self.nodes*2:3*self.nodes])-\
        (torch.matmul(self.get_dh(x),self.betas[0:self.nodes])**2\
        +torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])**2)**(1/2)\
        *torch.tan(torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes])+delta0)/self.l
        
        l_delta = torch.matmul(self.get_dh(x),self.betas[self.nodes*3:4*self.nodes])-self.rho
        loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        
        return loss

    def get_h(self,x):
        return torch.tanh(torch.add(torch.matmul(x,torch.transpose(self.W,0,1)),torch.transpose(self.b,0,1)))
    def get_dh(self,x):
        return torch.mul((1-self.get_h(x)**2),torch.transpose(self.W,0,1))