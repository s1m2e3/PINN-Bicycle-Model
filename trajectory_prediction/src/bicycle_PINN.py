import torch
from torch.autograd.functional import jacobian
import numpy as np
import pandas as pd
import datetime
import torch.nn as nn
import matplotlib.pyplot as plt

class Difference_RNN():
    def __init__(self,matrix_A_shape,matrix_B_shape,sequence_length):
        self.matrix_A = torch.randn(matrix_A_shape,dtype=torch.float,requires_grad=True)
        self.matrix_B = torch.randn(matrix_B_shape,dtype=torch.float,requires_grad=True)
        self.matrix_W = torch.randn(matrix_A_shape,dtype=torch.float)
        self.sequence_length= sequence_length
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def forward(self,x_0,u):
        """forward pass of difference equation, assume that the dimensions of the control u
          are given by [sequence_length, number of controls]"""
        x_out = torch.zeros((self.sequence_length, self.matrix_A.shape[0]), dtype=torch.float).to(self.device)
        for i in range(self.sequence_length):
            x_t = torch.matmul(x_0,self.matrix_A) + torch.matmul(u[:,i],self.matrix_B)
            x_out[i,:] = x_t
            x_0 = x_t
        return x_out
    def train(self,accuracy,n_iterations,x,u,y):
        optimizer = torch.optim.LBFGS([self.matrix_A,self.matrix_B])
        loss_history = []
        loss = nn.MSELoss()
        for i in range(n_iterations):
            optimizer.zero_grad()
            x_out = self.forward(x,u)
            output = loss(x_out,y)
            if output.item() < accuracy:
                break
            output.backward()
            loss_history.append(loss.item())
            optimizer.step()
        return loss_history

class Non_Linear_Difference_RNN(Difference_RNN):
    def __init__(self, matrix_A_shape, matrix_B_shape, sequence_length, hidden_size):
        super().__init__(matrix_A_shape, matrix_B_shape, sequence_length)
        self.matrix_A_shape = matrix_A_shape
        self.matrix_B_shape = matrix_B_shape
        self.x_to_WA_1 = torch.randn(matrix_A_shape[0],hidden_size, dtype=torch.float, requires_grad=True)
        self.b_WA_1 = torch.randn(1,hidden_size, dtype=torch.float, requires_grad=True)
        self.matrix_A = torch.randn(hidden_size,np.product(matrix_A_shape), dtype=torch.float, requires_grad=True)
        self.b_matrix_A = torch.randn(1,np.product(matrix_A_shape), dtype=torch.float, requires_grad=True)
        self.x_to_WB_1 = torch.randn(matrix_A_shape[0],hidden_size, dtype=torch.float, requires_grad=True)
        self.b_WB_1 = torch.randn(1,hidden_size, dtype=torch.float, requires_grad=True)
        self.matrix_B = torch.randn(hidden_size,np.product(matrix_B_shape), dtype=torch.float, requires_grad=True)
        self.b_matrix_B = torch.randn(1,np.product(matrix_B_shape), dtype=torch.float, requires_grad=True)            
        self.hidden_size = hidden_size
    
    def forward_matrices(self,x_0):
        x_0 = torch.matmul(x_0,self.x_to_WA_1) + self.b_WA_1
        x_0 = torch.tanh(x_0)
        matrix_A = torch.matmul(x_0,self.matrix_A) + self.b_matrix_A
        x_0 = torch.matmul(x_0,self.x_to_WB_1) + self.b_WB_1
        x_0 = torch.tanh(x_0)
        matrix_B = torch.matmul(x_0,self.matrix_B) + self.b_matrix_B
        return matrix_A.reshape(self.matrix_A_shape), matrix_B.reshape(self.matrix_B_shape)

    def forward(self,matrix_A,matrix_B,x_0,u):
        """forward pass of difference equation, assume that the dimensions of the control u
          are given by [sequence_length, number of controls]"""
        x_out = torch.zeros((self.sequence_length, matrix_A.shape[0]), dtype=torch.float).to(self.device)
        for i in range(self.sequence_length):
            x_t = torch.matmul(x_0,matrix_A) + torch.matmul(u[:,i],matrix_B)
            x_out[i,:] = x_t
            x_0 = x_t
        return x_out
    def train(self,accuracy,n_iterations,x,u,y):
        optimizer = torch.optim.LBFGS([self.matrix_A,self.matrix_B])
        loss_history = []
        loss = nn.MSELoss()
        for i in range(n_iterations):
            optimizer.zero_grad()
            matrix_A,matrix_B = self.forward_matrices(x)
            x_out = self.forward(matrix_A,matrix_B,x,u)
            output = loss(x_out,y)
            if output.item() < accuracy:
                break
            output.backward()
            loss_history.append(loss.item())
            optimizer.step()
        return loss_history

    def predict(self,x,u):
        matrix_A,matrix_B = self.forward_matrices(x)
        x_out = self.forward(matrix_A,matrix_B,x,u)
        return x_out
class PIELM:

    def __init__(self,n_nodes,input_size,output_size,length,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh",controls=False,physics=False):
        # if len(functions)==output_size:
        #     raise ValueError("gotta match number of states predicted and diferential equations")
        # self.functions = functions
        self.length= length
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nodes = n_nodes
        self.W = (torch.randn(size=(n_nodes,1),dtype=torch.float)*(high_w-low_w)+low_w)
        self.b = (torch.randn(size=(n_nodes,1),dtype=torch.float)*(high_b-low_b)+low_b)
        self.betas = torch.ones(size=(output_size*n_nodes,),requires_grad=True,dtype=torch.float)
        self.controls = controls
        self.physics = physics
        
    def train(self,accuracy, n_iterations,x_train,y_train,l,rho,steering_angle,slip_angle,speed_x,speed_y,heading_ratio,lambda_=1):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        count = 0
        error = 100
        self.lambda_ = lambda_
        
        z0 = -1
        zf = 1
        t0 = x_train[0]
        tf = x_train[-1]
        c = (zf-z0)/(tf-t0)
        x_train = z0+c*(x_train-t0)

        self.c = c
        self.x_train = torch.tensor(x_train,dtype=torch.float).reshape(x_train.shape[0],1)
        
        self.y_train = torch.tensor(y_train,dtype=torch.float)
        self.x_train_pred = self.x_train[:len(self.x_train)-self.length,]
        
        self.y_train_pred = self.y_train[:len(self.y_train)-self.length,]
        self.steering_angle = torch.tensor(steering_angle,dtype=torch.float)
        self.slip_angle = torch.tensor(slip_angle,dtype=torch.float)
        self.speed_x = torch.tensor(speed_x,dtype=torch.float)
        self.speed_y = torch.tensor(speed_y,dtype=torch.float)
        self.heading_ratio = torch.tensor(heading_ratio,dtype=torch.float)
        self.l = torch.tensor(l,dtype=torch.float)
        self.rho = torch.tensor(rho,dtype=torch.float)
        print(self.betas.is_cuda)
        print("number of samples:",len(self.x_train))
        while count < n_iterations:
            
            with torch.no_grad():
                
                jac = jacobian(self.predict_jacobian,self.betas)
                loss = self.predict_loss(self.x_train,self.y_train_pred,self.x_train_pred)
                pinv_jac = torch.linalg.pinv(jac)
                delta = torch.matmul(pinv_jac,loss)

                self.betas -=delta*0.1
            if count %10==0:
                print(loss.abs().max(dim=0),loss.mean(dim=0))
                #print(torch.mean(loss[0:4]))
                #print(torch.max(loss[0:4]))
                #print(torch.min(loss[0:4]))
                print("final loss:",(loss**2).mean())
                print(count)
            count +=1
        # print(loss[0:20],"x position")
        # print(loss[20:40],"y position")
        # print(loss[40:60],"angle")
        # print(loss[60:80],"steering")        
        # print(loss[80:100],"x speed")
        # print(loss[100:120],"y speed")
        # print(loss[120:140],"angular speed")
        # print(loss[140:160],"delta rate")        
        
        
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
        

        hx = torch.matmul(self.get_h(self.x_train_pred),betas[0:self.nodes])
        hy = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes:2*self.nodes])
        htheta = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*2:3*self.nodes])
        htheta_full = torch.matmul(self.get_h(self.x_train),betas[self.nodes*2:3*self.nodes])
        # hdelta = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*3:4*self.nodes])
        # hdelta_full = torch.matmul(self.get_h(self.x_train),betas[self.nodes*3:4*self.nodes])
        l_pred_x = self.y_train_pred[:,0]-hx
        l_pred_y = self.y_train_pred[:,1]-hy
        l_pred_theta = self.y_train_pred[:,2]-htheta
        # l_pred_delta = self.y_train_pred[:,3]-hdelta
        #l_pred_delta = self.y_train[:,3]-torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*3:4*self.nodes])
        
        dhx = self.c*torch.matmul(self.get_dh(self.x_train),betas[0:self.nodes])
        dhy = self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes:2*self.nodes])
        dhtheta = self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes*2:3*self.nodes])
        # dhdelta = self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes*3:4*self.nodes])

        l_x = dhx-(dhx**2+dhy**2)**(1/2)*torch.cos(htheta_full+self.slip_angle)
        l_y = dhy-(dhx**2+dhy**2)**(1/2)*torch.sin(htheta_full+self.slip_angle)
        l_theta = dhtheta-(dhx**2+dhy**2)**(1/2)*torch.tan(self.steering_angle)*torch.cos(self.slip_angle)/self.l


        # l_x = dhx-(dhx**2+dhy**2)**(1/2)*torch.cos(htheta_full)
        # l_y = dhy-(dhx**2+dhy**2)**(1/2)*torch.sin(htheta_full)
        # l_theta = dhtheta-(dhx**2+dhy**2)**(1/2)*torch.tan(hdelta_full)/self.l
        # l_delta = dhdelta-self.rho
        
        l_pred_dhx = self.speed_x- dhx
        l_pred_dhy = self.speed_y- dhy 
        l_pred_dhtheta = self.heading_ratio - dhtheta
          
        #loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        # loss= torch.hstack((l_pred_x,l_pred_y,l_x,l_y))
        if self.controls and self.physics:
            l_pred_dhtheta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_pred_dhtheta[0:len(self.x_train_pred)]
            l_pred_dhtheta[len(self.x_train_pred):]=self.lambda_*l_pred_dhtheta[len(self.x_train_pred):]
            l_theta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_theta[0:len(self.x_train_pred)]
            l_theta[len(self.x_train_pred):]=self.lambda_*l_theta[len(self.x_train_pred):]
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                                self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,self.lambda_*l_pred_dhtheta,\
                                (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,l_theta))
            
        if self.controls and not self.physics:
            l_pred_dhtheta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_pred_dhtheta[0:len(self.x_train_pred)]
            l_pred_dhtheta[len(self.x_train_pred):]=self.lambda_*l_pred_dhtheta[len(self.x_train_pred):]
            self.lambda_ = 1
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                                self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,self.lambda_*l_pred_dhtheta))
            
        if not self.controls and not self.physics:
            self.lambda_ = 1
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta))
                                
        return loss
            
    def predict_loss(self,x,y,x_pred):
       

        hx = torch.matmul(self.get_h(x_pred),self.betas[0:self.nodes])
        hy = torch.matmul(self.get_h(x_pred),self.betas[self.nodes:2*self.nodes])
        htheta = torch.matmul(self.get_h(x_pred),self.betas[self.nodes*2:3*self.nodes])
        htheta_full = torch.matmul(self.get_h(self.x_train),self.betas[self.nodes*2:3*self.nodes])
        # hdelta = torch.matmul(self.get_h(self.x_train_pred),self.betas[self.nodes*3:4*self.nodes])
        # hdelta_full = torch.matmul(self.get_h(self.x_train),self.betas[self.nodes*3:4*self.nodes])
        l_pred_x = y[:,0]-hx
        l_pred_y = y[:,1]-hy
        l_pred_theta = y[:,2]-htheta
        # l_pred_delta = self.y_train_pred[:,3]-hdelta
        #l_pred_delta = self.y_train[:,3]-torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*3:4*self.nodes])
        
        dhx = self.c*torch.matmul(self.get_dh(x),self.betas[0:self.nodes])
        dhy = self.c*torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])
        dhtheta = self.c*torch.matmul(self.get_dh(x),self.betas[self.nodes*2:3*self.nodes])
        # dhdelta = self.c*torch.matmul(self.get_dh(x),self.betas[self.nodes*3:4*self.nodes])
    
        l_x = dhx-(dhx**2+dhy**2)**(1/2)*torch.cos(htheta_full+self.slip_angle)
        l_y = dhy-(dhx**2+dhy**2)**(1/2)*torch.sin(htheta_full+self.slip_angle)
        l_theta = dhtheta-(dhx**2+dhy**2)**(1/2)*torch.tan(self.steering_angle)*torch.cos(self.slip_angle)/self.l
        
        # l_x = dhx-(dhx**2+dhy**2)**(1/2)*torch.cos(htheta_full)
        # l_y = dhy-(dhx**2+dhy**2)**(1/2)*torch.sin(htheta_full)
        # l_theta = dhtheta-(dhx**2+dhy**2)**(1/2)*torch.tan(hdelta_full)/self.l
        l_pred_dhtheta = self.heading_ratio - dhtheta
        l_pred_dhx = self.speed_x- dhx
        l_pred_dhy = self.speed_y- dhy 
        # l_delta = dhdelta-self.rho
        
        #loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        # loss= torch.hstack((l_pred_x,l_pred_y,l_x,l_y))
        # loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
        #                     self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,\
        #                     (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))
        if self.controls and self.physics:
            l_pred_dhtheta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_pred_dhtheta[0:len(self.x_train_pred)]
            l_pred_dhtheta[len(self.x_train_pred):]=self.lambda_*l_pred_dhtheta[len(self.x_train_pred):]
            l_theta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_theta[0:len(self.x_train_pred)]
            l_theta[len(self.x_train_pred):]=self.lambda_*l_theta[len(self.x_train_pred):]
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                                self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,self.lambda_*l_pred_dhtheta,\
                                (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,l_theta))
        if self.controls and not self.physics:
            l_pred_dhtheta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_pred_dhtheta[0:len(self.x_train_pred)]
            l_pred_dhtheta[len(self.x_train_pred):]=self.lambda_*l_pred_dhtheta[len(self.x_train_pred):]
            self.lambda_ = 1
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                                self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,self.lambda_*l_pred_dhtheta))
        if not self.controls and not self.physics:
            self.lambda_ = 1
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta))

        return loss    
    
    def get_h(self,x):
        return torch.tanh(torch.add(torch.matmul(x,torch.transpose(self.W,0,1)),torch.transpose(self.b,0,1)))
    def get_dh(self,x):
        return torch.mul((1-self.get_h(x)**2),torch.transpose(self.W,0,1))
    
    def pred(self,x):
        
        z0 = -1
        t0 = x[0]
        x = z0+self.c*(x-t0)

        x = torch.tensor(np.array(x),dtype=torch.float).reshape(x.shape[0],1)
        
        x_pred = torch.matmul(self.get_h(x),self.betas[0:self.nodes]) 
        y_pred = torch.matmul(self.get_h(x),self.betas[self.nodes:2*self.nodes])
        theta_pred = torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes])
        delta_pred = torch.matmul(self.get_h(x),self.betas[self.nodes*3:4*self.nodes])
        return torch.vstack((x_pred,y_pred,theta_pred,delta_pred))

class XTFC(PIELM):
    def __init__(self,n_nodes,input_size,output_size,length,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh",controls=False,physics=False):
        super().__init__(n_nodes,input_size,output_size,length,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh",controls=False,physics=False)
        self.length= length
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nodes = n_nodes
        self.W = (torch.randn(size=(n_nodes,1),dtype=torch.float)*(high_w-low_w)+low_w)
        self.b = (torch.randn(size=(n_nodes,1),dtype=torch.float)*(high_b-low_b)+low_b)
        self.betas = torch.ones(size=(output_size*n_nodes,),requires_grad=True,dtype=torch.float)
        self.controls = controls
        self.physics = physics

    def predict_jacobian(self,betas):
        
        """""
        For a 4 point constrained problem on x and y  we have: 

        Consider support functions the polynomials: t^0,t^1,t^2,t^3: Then for x and y we have:
        x(0)       [[1,0,0,0]
        x(f)       [1,t,t^2,t^3]
        xdot(0)    [0,1,0,0]
        xdotdot(0) [0,0,2,0]]

        Consider support functions the polynomials: t^0,t^1,t^2,t^3,t^4,t^5: Then for x and y we have:
        x(0)       [[1,t(0),t(0)^2,t(0)^3,t(0)^4,t(0)^5]
        x(r1)      [1,t(r1),t(r1)^2,t(r1)^3,t(r1)^4,t(r1)^5]
        x(r3)      [1,t(r3),t(r3)^2,t(r3)^3,t(r3)^4,t(r3)^5]
        x(f)       [1,t(rf),t(rf)^2,t(rf)^3]
        xdot(0)    [0,1,0,0]
        xdotdot(0) [0,0,2,0]]
        

        For a 2 point constrained problem on theta we have:
        Consider support functions the polynomials: t^0,t^1: Then for theta have:
        theta(0)       [[1,0]
        theta(f)       [1,t]
        """""
        h = self.get_h(self.x_train)
        dh = self.get_dh(self.x_train)
        bx = betas[0:self.nodes]
        by = betas[self.nodes:self.nodes*2]
        btheta = betas[self.nodes*2:self.nodes*3]
        
        init_time = self.x_train[0].numpy()[0]
        final_time_pred = self.x_train_pred[-1].numpy()[0]
        final_time_total = self.x_train[-1].numpy()[0]
        init_h=self.get_h(self.x_train[0])
        init_dh=self.get_dh(self.x_train[0])
        final_pred_h=self.get_h(self.x_train_pred[-1])
        final_pred_dh=self.get_dh(self.x_train_pred[-1])
        final_dh=self.get_dh(self.x_train[-1])
        init_x = self.y_train[0,0]
        init_y = self.y_train[0,1]
        init_theta = self.y_train[0,2]
        final_pred_x = self.y_train[len(self.y_train_pred)-1,0]
        final_pred_y = self.y_train[len(self.y_train_pred)-1,1]
        final_pred_theta = self.y_train[len(self.y_train_pred)-1,2]
        init_dx = self.speed_x[0]
        init_dy = self.speed_y[0]
        init_dtheta = self.heading_ratio[0]
        final_pred_dx = self.speed_x[len(self.y_train_pred)-1]
        final_pred_dy = self.speed_y[len(self.y_train_pred)-1]
        final_pred_dtheta = self.heading_ratio[len(self.y_train_pred)-1]
        final_total_dx = self.speed_x[-1]
        final_total_dy = self.speed_y[-1]
        final_total_dtheta = self.heading_ratio[-1]
        
        # support_function_matrix = np.array([[1,init_time,init_time**2,init_time**3,init_time**4],\
                                            
        #                                     [1,final_time_pred,final_time_pred**2,final_time_pred**3,final_time_pred**4],\
                                            
        #                                     [0,1,2*init_time,3*init_time**2,4*init_time**3],\
                                            
        #                                     [0,1,2*final_time_pred,3*final_time_pred**2,4*final_time_pred**3],\
                                            
        #                                     [0,1,2*final_time_total,3*final_time_total**2,4*final_time_total**3]])        
        support_function_matrix = np.array([[1,init_time,init_time**2,init_time**3],\
                                            
                                            [1,final_time_pred,final_time_pred**2,final_time_pred**3],\
                                            
                                            [0,1,2*init_time,3*init_time**2],\
                                            
                                            [0,1,2*final_time_pred,3*final_time_pred**2]])        
        
        coefficients_matrix = torch.tensor(np.linalg.inv(support_function_matrix),dtype=torch.float)
        
        # free_support_function_matrix = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train,self.x_train**2,self.x_train**3,self.x_train**4))
        # d_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape),2*self.x_train,3*self.x_train**2,4*self.x_train**3))
        
        free_support_function_matrix = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train,self.x_train**2,self.x_train**3))
        d_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape),2*self.x_train,3*self.x_train**2))
        


        phis = torch.matmul(free_support_function_matrix,coefficients_matrix)
        phi1 = phis[:,0].reshape(len(self.x_train),1)
        phi2 = phis[:,1].reshape(len(self.x_train),1)
        phi3 = phis[:,2].reshape(len(self.x_train),1)
        phi4 = phis[:,3].reshape(len(self.x_train),1)
        # phi5 = phis[:,4].reshape(len(self.x_train),1)
        d_phis = torch.matmul(d_free_support_function_matrix,coefficients_matrix)
        d_phi1 = d_phis[:,0].reshape(len(self.x_train),1)
        d_phi2 = d_phis[:,1].reshape(len(self.x_train),1)
        d_phi3 = d_phis[:,2].reshape(len(self.x_train),1)
        d_phi4 = d_phis[:,3].reshape(len(self.x_train),1)
        # d_phi5 = d_phis[:,4].reshape(len(self.x_train),1)
        
        
        

        phi1_h_init = torch.matmul(-phi1,init_h)
        phi1_x_init = phi1*init_x
        phi1_y_init = phi1*init_y
        phi1_theta_init = phi1*init_theta
        
        phi2_h_predf = torch.matmul(-phi2,final_pred_h)
        phi2_x_predf = phi2*final_pred_x
        phi2_y_predf = phi2*final_pred_y
        phi2_theta_predf = phi2*final_pred_theta

        phi3_dh_init = torch.matmul(-phi3,init_dh)
        phi3_dx_init = phi3*init_dx
        phi3_dy_init = phi3*init_dy
        phi3_dtheta_init = phi3*init_dtheta

        phi4_dh_predf = torch.matmul(-phi4,final_pred_dh)
        phi4_dx_predf = phi4*final_pred_dx
        phi4_dy_predf = phi4*final_pred_dy
        phi4_dtheta_predf = phi4*final_pred_dtheta

        # phi5_dh_final = torch.matmul(-phi5,final_dh)
        # phi5_dx_final = phi5*final_total_dx
        # phi5_dy_final = phi5*final_total_dy
        # phi5_dtheta_final = phi5*final_total_dtheta

        dphi1_h_init = torch.matmul(-d_phi1,init_h)
        dphi1_x_init = d_phi1*init_x
        dphi1_y_init = d_phi1*init_y
        dphi1_theta_init = d_phi1*init_theta
        
        dphi2_h_predf = torch.matmul(-d_phi2,final_pred_h)
        dphi2_x_predf = d_phi2*final_pred_x
        dphi2_y_predf = d_phi2*final_pred_y
        dphi2_theta_predf = d_phi2*final_pred_theta

        dphi3_dh_init = torch.matmul(-d_phi3,init_dh)
        dphi3_dx_init = d_phi3*init_dx
        dphi3_dy_init = d_phi3*init_dy
        dphi3_dtheta_init = d_phi3*init_dtheta

        dphi4_dh_predf = torch.matmul(-d_phi4,final_pred_dh)
        dphi4_dx_predf = d_phi4*final_pred_dx
        dphi4_dy_predf = d_phi4*final_pred_dy
        dphi4_dtheta_predf = d_phi4*final_pred_dtheta

        # dphi5_dh_final = torch.matmul(-d_phi5,final_dh)
        # dphi5_dx_final = d_phi5*final_total_dx
        # dphi5_dy_final = d_phi5*final_total_dy
        # dphi5_dtheta_final = d_phi5*final_total_dtheta


        
        # hx = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf).add(phi5_dh_final),bx).reshape(self.x_train.shape)\
        # .add(phi1_x_init).add(phi2_x_predf).add(phi3_dx_init/self.c).add(phi4_dx_predf/self.c).add(phi5_dx_final/self.c))[:,0]
           
        # dhx = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf).add(dphi5_dh_final),bx).reshape(self.x_train.shape)\
        # .add(dphi1_x_init).add(dphi2_x_predf).add(dphi3_dx_init/self.c).add(dphi4_dx_predf/self.c).add(dphi5_dx_final/self.c))[:,0]
       
        # hy = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf).add(phi5_dh_final),by).reshape(self.x_train.shape)\
        # .add(phi1_y_init).add(phi2_y_predf).add(phi3_dy_init/self.c).add(phi4_dy_predf/self.c).add(phi5_dy_final/self.c))[:,0]
        
        # dhy = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf).add(dphi5_dh_final),by).reshape(self.x_train.shape)\
        # .add(dphi1_y_init).add(dphi2_y_predf).add(dphi3_dy_init/self.c).add(dphi4_dy_predf/self.c).add(dphi5_dy_final/self.c))[:,0]

        # htheta = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf).add(phi5_dh_final),btheta).reshape(self.x_train.shape)\
        # .add(phi1_theta_init).add(phi2_theta_predf).add(phi3_dtheta_init/self.c).add(phi4_dtheta_predf/self.c).add(phi5_dtheta_final/self.c))[:,0]
     
        # dhtheta = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf).add(dphi5_dh_final),btheta).reshape(self.x_train.shape)\
        # .add(dphi1_theta_init).add(dphi2_theta_predf).add(dphi3_dtheta_init/self.c).add(dphi4_dtheta_predf/self.c).add(dphi5_dtheta_final/self.c))[:,0]

        hx = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf),bx).reshape(self.x_train.shape)\
        .add(phi1_x_init).add(phi2_x_predf).add(phi3_dx_init/self.c).add(phi4_dx_predf/self.c))[:,0]
           
        dhx = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf),bx).reshape(self.x_train.shape)\
        .add(dphi1_x_init).add(dphi2_x_predf).add(dphi3_dx_init/self.c).add(dphi4_dx_predf/self.c))[:,0]
       
        hy = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf),by).reshape(self.x_train.shape)\
        .add(phi1_y_init).add(phi2_y_predf).add(phi3_dy_init/self.c).add(phi4_dy_predf/self.c))[:,0]
        
        dhy = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf),by).reshape(self.x_train.shape)\
        .add(dphi1_y_init).add(dphi2_y_predf).add(dphi3_dy_init/self.c).add(dphi4_dy_predf/self.c))[:,0]

        htheta = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf),btheta).reshape(self.x_train.shape)\
        .add(phi1_theta_init).add(phi2_theta_predf).add(phi3_dtheta_init/self.c).add(phi4_dtheta_predf/self.c))[:,0]
     
        dhtheta = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf),btheta).reshape(self.x_train.shape)\
        .add(dphi1_theta_init).add(dphi2_theta_predf).add(dphi3_dtheta_init/self.c).add(dphi4_dtheta_predf/self.c))[:,0]

        l_pred_x = self.y_train_pred[:,0]-hx[0:len(self.x_train_pred)]
        
        l_pred_y = self.y_train_pred[:,1]-hy[0:len(self.x_train_pred)]
        l_pred_theta = self.y_train_pred[:,2]-htheta[0:len(self.x_train_pred)]
        
        l_x = dhx-(((dhx)**2+ (dhy)**2)**(1/2)*torch.cos(htheta+self.slip_angle))
       
        l_y = dhy-(((dhx)**2+ (dhy)**2)**(1/2)*torch.sin(htheta+self.slip_angle)) 
        l_theta = dhtheta - (((dhx)**2+ (dhy)**2)**(1/2))*torch.tan(self.steering_angle)*torch.cos(self.slip_angle)/self.l
        l_pred_dhx = self.speed_x - dhx
        l_pred_dhy = self.speed_y - dhy
        l_pred_dhtheta = self.heading_ratio - dhtheta 
        

        if self.controls and self.physics:
            l_pred_dhtheta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_pred_dhtheta[0:len(self.x_train_pred)]
            l_pred_dhtheta[len(self.x_train_pred):]=self.lambda_*l_pred_dhtheta[len(self.x_train_pred):]
            l_theta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_theta[0:len(self.x_train_pred)]
            l_theta[len(self.x_train_pred):]=self.lambda_*l_theta[len(self.x_train_pred):]
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                                self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,self.lambda_*l_pred_dhtheta,\
                                (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,l_theta))
            # loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
            #                     self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,\
            #                     (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))
            
        if self.controls and not self.physics:
            l_pred_dhtheta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_pred_dhtheta[0:len(self.x_train_pred)]
            l_pred_dhtheta[len(self.x_train_pred):]=self.lambda_*l_pred_dhtheta[len(self.x_train_pred):]
            self.lambda_ = 1
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                                self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,self.lambda_*l_pred_dhtheta))
            # loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
            #                     self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy))
            
        if not self.controls and not self.physics:
           
            self.lambda_ = 1
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta))

        
        return loss
            
    def predict_loss(self,x,y,x_pred):
         
        """""
        For a 4 point constrained problem on x and y  we have: 

        Consider support functions the polynomials: t^0,t^1,t^2,t^3: Then for x and y we have:
        x(0)       [[1,0,0,0]
        x(f)       [1,t,t^2,t^3]
        xdot(0)    [0,1,0,0]
        xdotdot(0) [0,0,2,0]]

        Consider support functions the polynomials: t^0,t^1,t^2,t^3,t^4,t^5: Then for x and y we have:
        x(0)       [[1,t(0),t(0)^2,t(0)^3,t(0)^4,t(0)^5]
        x(r1)      [1,t(r1),t(r1)^2,t(r1)^3,t(r1)^4,t(r1)^5]
        x(r3)      [1,t(r3),t(r3)^2,t(r3)^3,t(r3)^4,t(r3)^5]
        x(f)       [1,t(rf),t(rf)^2,t(rf)^3]
        xdot(0)    [0,1,0,0]
        xdotdot(0) [0,0,2,0]]
        

        For a 2 point constrained problem on theta we have:
        Consider support functions the polynomials: t^0,t^1: Then for theta have:
        theta(0)       [[1,0]
        theta(f)       [1,t]
        """""
        h = self.get_h(x)
        dh = self.get_dh(x)
        bx = self.betas[0:self.nodes]
        by = self.betas[self.nodes:self.nodes*2]
        btheta = self.betas[self.nodes*2:self.nodes*3]

        init_time = self.x_train[0].numpy()[0]
        final_time_pred = self.x_train_pred[-1].numpy()[0]
        final_time_total = self.x_train[len(self.x_train)-1].numpy()[0]
        init_h=self.get_h(self.x_train[0])
        init_dh=self.get_dh(self.x_train[0])
        final_pred_h=self.get_h(self.x_train_pred[-1])
        final_pred_dh=self.get_dh(self.x_train_pred[-1])
        final_dh=self.get_dh(self.x_train[-1])
        init_x = self.y_train[0,0]
        init_y = self.y_train[0,1]
        init_theta = self.y_train[0,2]
        init_dx = self.speed_x[0]
        init_dy = self.speed_y[0]
        init_dtheta = self.heading_ratio[0]
        final_pred_x = self.y_train[len(self.y_train_pred)-1,0]
        final_pred_y = self.y_train[len(self.y_train_pred)-1,1]
        final_pred_theta = self.y_train[len(self.y_train_pred)-1,2]
        final_pred_dx = self.speed_x[len(self.y_train_pred)-1]
        final_pred_dy = self.speed_y[len(self.y_train_pred)-1]
        final_pred_dtheta = self.heading_ratio[len(self.y_train_pred)-1]
        final_total_dx = self.speed_x[-1]
        final_total_dy = self.speed_y[-1]
        final_total_dtheta = self.heading_ratio[-1]
        
        # support_function_matrix = np.array([[1,init_time,init_time**2,init_time**3,init_time**4],\
                                            
        #                                     [1,final_time_pred,final_time_pred**2,final_time_pred**3,final_time_pred**4],\
                                            
        #                                     [0,1,2*init_time,3*init_time**2,4*init_time**3],\
                                            
        #                                     [0,1,2*final_time_pred,3*final_time_pred**2,4*final_time_pred**3],\
                                            
        #                                     [0,1,2*final_time_total,3*final_time_total**2,4*final_time_total**3]])        
        support_function_matrix = np.array([[1,init_time,init_time**2,init_time**3],\
                                            
                                            [1,final_time_pred,final_time_pred**2,final_time_pred**3],\
                                            
                                            [0,1,2*init_time,3*init_time**2],\
                                            
                                            [0,1,2*final_time_pred,3*final_time_pred**2]])        
        
        coefficients_matrix = torch.tensor(np.linalg.inv(support_function_matrix),dtype=torch.float)
        
        # free_support_function_matrix = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train,self.x_train**2,self.x_train**3,self.x_train**4))
        # d_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape),2*self.x_train,3*self.x_train**2,4*self.x_train**3))
        
        free_support_function_matrix = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train,self.x_train**2,self.x_train**3))
        d_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape),2*self.x_train,3*self.x_train**2))
        


        phis = torch.matmul(free_support_function_matrix,coefficients_matrix)
        phi1 = phis[:,0].reshape(len(self.x_train),1)
        phi2 = phis[:,1].reshape(len(self.x_train),1)
        phi3 = phis[:,2].reshape(len(self.x_train),1)
        phi4 = phis[:,3].reshape(len(self.x_train),1)
        # phi5 = phis[:,4].reshape(len(self.x_train),1)
        d_phis = torch.matmul(d_free_support_function_matrix,coefficients_matrix)
        d_phi1 = d_phis[:,0].reshape(len(self.x_train),1)
        d_phi2 = d_phis[:,1].reshape(len(self.x_train),1)
        d_phi3 = d_phis[:,2].reshape(len(self.x_train),1)
        d_phi4 = d_phis[:,3].reshape(len(self.x_train),1)
        # d_phi5 = d_phis[:,4].reshape(len(self.x_train),1)
        
        
        

        phi1_h_init = torch.matmul(-phi1,init_h)
        phi1_x_init = phi1*init_x
        phi1_y_init = phi1*init_y
        phi1_theta_init = phi1*init_theta
        
        phi2_h_predf = torch.matmul(-phi2,final_pred_h)
        phi2_x_predf = phi2*final_pred_x
        phi2_y_predf = phi2*final_pred_y
        phi2_theta_predf = phi2*final_pred_theta

        phi3_dh_init = torch.matmul(-phi3,init_dh)
        phi3_dx_init = phi3*init_dx
        phi3_dy_init = phi3*init_dy
        phi3_dtheta_init = phi3*init_dtheta

        phi4_dh_predf = torch.matmul(-phi4,final_pred_dh)
        phi4_dx_predf = phi4*final_pred_dx
        phi4_dy_predf = phi4*final_pred_dy
        phi4_dtheta_predf = phi4*final_pred_dtheta

        # phi5_dh_final = torch.matmul(-phi5,final_dh)
        # phi5_dx_final = phi5*final_total_dx
        # phi5_dy_final = phi5*final_total_dy
        # phi5_dtheta_final = phi5*final_total_dtheta

        dphi1_h_init = torch.matmul(-d_phi1,init_h)
        dphi1_x_init = d_phi1*init_x
        dphi1_y_init = d_phi1*init_y
        dphi1_theta_init = d_phi1*init_theta
        
        dphi2_h_predf = torch.matmul(-d_phi2,final_pred_h)
        dphi2_x_predf = d_phi2*final_pred_x
        dphi2_y_predf = d_phi2*final_pred_y
        dphi2_theta_predf = d_phi2*final_pred_theta

        dphi3_dh_init = torch.matmul(-d_phi3,init_dh)
        dphi3_dx_init = d_phi3*init_dx
        dphi3_dy_init = d_phi3*init_dy
        dphi3_dtheta_init = d_phi3*init_dtheta

        dphi4_dh_predf = torch.matmul(-d_phi4,final_pred_dh)
        dphi4_dx_predf = d_phi4*final_pred_dx
        dphi4_dy_predf = d_phi4*final_pred_dy
        dphi4_dtheta_predf = d_phi4*final_pred_dtheta

        # dphi5_dh_final = torch.matmul(-d_phi5,final_dh)
        # dphi5_dx_final = d_phi5*final_total_dx
        # dphi5_dy_final = d_phi5*final_total_dy
        # dphi5_dtheta_final = d_phi5*final_total_dtheta


        
        # hx = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf).add(phi5_dh_final),bx).reshape(self.x_train.shape)\
        # .add(phi1_x_init).add(phi2_x_predf).add(phi3_dx_init/self.c).add(phi4_dx_predf/self.c).add(phi5_dx_final/self.c))[:,0]
           
        # dhx = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf).add(dphi5_dh_final),bx).reshape(self.x_train.shape)\
        # .add(dphi1_x_init).add(dphi2_x_predf).add(dphi3_dx_init/self.c).add(dphi4_dx_predf/self.c).add(dphi5_dx_final/self.c))[:,0]
       
        # hy = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf).add(phi5_dh_final),by).reshape(self.x_train.shape)\
        # .add(phi1_y_init).add(phi2_y_predf).add(phi3_dy_init/self.c).add(phi4_dy_predf/self.c).add(phi5_dy_final/self.c))[:,0]
        
        # dhy = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf).add(dphi5_dh_final),by).reshape(self.x_train.shape)\
        # .add(dphi1_y_init).add(dphi2_y_predf).add(dphi3_dy_init/self.c).add(dphi4_dy_predf/self.c).add(dphi5_dy_final/self.c))[:,0]

        # htheta = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf).add(phi5_dh_final),btheta).reshape(self.x_train.shape)\
        # .add(phi1_theta_init).add(phi2_theta_predf).add(phi3_dtheta_init/self.c).add(phi4_dtheta_predf/self.c).add(phi5_dtheta_final/self.c))[:,0]
     
        # dhtheta = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf).add(dphi5_dh_final),btheta).reshape(self.x_train.shape)\
        # .add(dphi1_theta_init).add(dphi2_theta_predf).add(dphi3_dtheta_init/self.c).add(dphi4_dtheta_predf/self.c).add(dphi5_dtheta_final/self.c))[:,0]

        hx = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf),bx).reshape(self.x_train.shape)\
        .add(phi1_x_init).add(phi2_x_predf).add(phi3_dx_init/self.c).add(phi4_dx_predf/self.c))[:,0]
           
        dhx = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf),bx).reshape(self.x_train.shape)\
        .add(dphi1_x_init).add(dphi2_x_predf).add(dphi3_dx_init/self.c).add(dphi4_dx_predf/self.c))[:,0]
       
        hy = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf),by).reshape(self.x_train.shape)\
        .add(phi1_y_init).add(phi2_y_predf).add(phi3_dy_init/self.c).add(phi4_dy_predf/self.c))[:,0]
        
        dhy = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf),by).reshape(self.x_train.shape)\
        .add(dphi1_y_init).add(dphi2_y_predf).add(dphi3_dy_init/self.c).add(dphi4_dy_predf/self.c))[:,0]

        htheta = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf),btheta).reshape(self.x_train.shape)\
        .add(phi1_theta_init).add(phi2_theta_predf).add(phi3_dtheta_init/self.c).add(phi4_dtheta_predf/self.c))[:,0]
     
        dhtheta = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf),btheta).reshape(self.x_train.shape)\
        .add(dphi1_theta_init).add(dphi2_theta_predf).add(dphi3_dtheta_init/self.c).add(dphi4_dtheta_predf/self.c))[:,0]
        
        
        l_pred_x = self.y_train_pred[:,0]-hx[0:len(self.x_train_pred)]
        
        l_pred_y = self.y_train_pred[:,1]-hy[0:len(self.x_train_pred)]
        l_pred_theta = self.y_train_pred[:,2]-htheta[0:len(self.x_train_pred)]
        
        l_x = dhx-(((dhx)**2+ (dhy)**2)**(1/2)*torch.cos(htheta+self.slip_angle))
       
        l_y = dhy-(((dhx)**2+ (dhy)**2)**(1/2)*torch.sin(htheta+self.slip_angle)) 
        l_theta = dhtheta - (((dhx)**2+ (dhy)**2)**(1/2))*torch.tan(self.steering_angle)*torch.cos(self.slip_angle)/self.l
        
        l_pred_dhx = self.speed_x - dhx
        l_pred_dhy = self.speed_y - dhy
        l_pred_dhtheta = self.heading_ratio - dhtheta 
        
        if self.controls and self.physics:
            l_pred_dhtheta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_pred_dhtheta[0:len(self.x_train_pred)]
            l_pred_dhtheta[len(self.x_train_pred):]=self.lambda_*l_pred_dhtheta[len(self.x_train_pred):]
            l_theta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_theta[0:len(self.x_train_pred)]
            l_theta[len(self.x_train_pred):]=self.lambda_*l_theta[len(self.x_train_pred):]
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                                self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,l_pred_dhtheta,\
                                (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,l_theta))
            # loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
            #                     self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,\
            #                     (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))
            
        if self.controls and not self.physics:
            
            self.lambda_ = 1
            l_pred_dhtheta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_pred_dhtheta[0:len(self.x_train_pred)]
            l_pred_dhtheta[len(self.x_train_pred):]=self.lambda_*l_pred_dhtheta[len(self.x_train_pred):]
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                                self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,l_pred_dhtheta))
            # loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
            #                     self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy))
            
        if not self.controls and not self.physics:
           
            self.lambda_ = 1
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta))
                                

        
        return loss
    
    def get_h(self,x):
        return torch.tanh(torch.add(torch.matmul(x,torch.transpose(self.W,0,1)),torch.transpose(self.b,0,1)))
    def get_dh(self,x):
        return torch.mul((1-self.get_h(x)**2),torch.transpose(self.W,0,1))
    
    def pred(self,x):


        z0 = -1
        zf = 1
        t0 = x[0]
        tf = x[-1]
        c = (zf-z0)/(tf-t0)
        x = z0+c*(x-t0)
        x = torch.tensor(np.array(x),dtype=torch.float).reshape(x.shape[0],1)
        

        """""
        For a 4 point constrained problem on x and y  we have: 

        Consider support functions the polynomials: t^0,t^1,t^2,t^3: Then for x and y we have:
        x(0)       [[1,0,0,0]
        x(f)       [1,t,t^2,t^3]
        xdot(0)    [0,1,0,0]
        xdotdot(0) [0,0,2,0]]

        Consider support functions the polynomials: t^0,t^1,t^2,t^3,t^4,t^5: Then for x and y we have:
        x(0)       [[1,t(0),t(0)^2,t(0)^3,t(0)^4,t(0)^5]
        x(r1)      [1,t(r1),t(r1)^2,t(r1)^3,t(r1)^4,t(r1)^5]
        x(r3)      [1,t(r3),t(r3)^2,t(r3)^3,t(r3)^4,t(r3)^5]
        x(f)       [1,t(rf),t(rf)^2,t(rf)^3]
        xdot(0)    [0,1,0,0]
        xdotdot(0) [0,0,2,0]]
        

        For a 2 point constrained problem on theta we have:
        Consider support functions the polynomials: t^0,t^1: Then for theta have:
        theta(0)       [[1,0]
        theta(f)       [1,t]
        """""
        h = self.get_h(x)
        dh = self.get_dh(x)
        bx = self.betas[0:self.nodes]
        by = self.betas[self.nodes:self.nodes*2]
        btheta = self.betas[self.nodes*2:self.nodes*3]

        init_time = self.x_train[0].numpy()[0]
        final_time_pred = self.x_train_pred[-1].numpy()[0]
        final_time_total = self.x_train[len(self.x_train)-1].numpy()[0]
        init_h=self.get_h(self.x_train[0])
        init_dh=self.get_dh(self.x_train[0])
        final_pred_h=self.get_h(self.x_train_pred[-1])
        final_pred_dh=self.get_dh(self.x_train_pred[-1])
        final_dh=self.get_dh(self.x_train[-1])
        init_x = self.y_train[0,0]
        init_y = self.y_train[0,1]
        init_theta = self.y_train[0,2]
        init_dx = self.speed_x[0]
        init_dy = self.speed_y[0]
        init_dtheta = self.heading_ratio[0]
        final_pred_x = self.y_train[len(self.y_train_pred)-1,0]
        final_pred_y = self.y_train[len(self.y_train_pred)-1,1]
        final_pred_theta = self.y_train[len(self.y_train_pred)-1,2]
        final_pred_dx = self.speed_x[len(self.y_train_pred)-1]
        final_pred_dy = self.speed_y[len(self.y_train_pred)-1]
        final_pred_dtheta = self.heading_ratio[len(self.y_train_pred)-1]
        final_total_dx = self.speed_x[-1]
        final_total_dy = self.speed_y[-1]
        final_total_dtheta = self.heading_ratio[-1]
        

         # support_function_matrix = np.array([[1,init_time,init_time**2,init_time**3,init_time**4],\
                                            
        #                                     [1,final_time_pred,final_time_pred**2,final_time_pred**3,final_time_pred**4],\
                                            
        #                                     [0,1,2*init_time,3*init_time**2,4*init_time**3],\
                                            
        #                                     [0,1,2*final_time_pred,3*final_time_pred**2,4*final_time_pred**3],\
                                            
        #                                     [0,1,2*final_time_total,3*final_time_total**2,4*final_time_total**3]])        
        support_function_matrix = np.array([[1,init_time,init_time**2,init_time**3],\
                                            
                                            [1,final_time_pred,final_time_pred**2,final_time_pred**3],\
                                            
                                            [0,1,2*init_time,3*init_time**2],\
                                            
                                            [0,1,2*final_time_pred,3*final_time_pred**2]])        
        
        coefficients_matrix = torch.tensor(np.linalg.inv(support_function_matrix),dtype=torch.float)
        
        # free_support_function_matrix = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train,self.x_train**2,self.x_train**3,self.x_train**4))
        # d_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape),2*self.x_train,3*self.x_train**2,4*self.x_train**3))
        
        free_support_function_matrix = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train,self.x_train**2,self.x_train**3))
        d_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape),2*self.x_train,3*self.x_train**2))
 
        phis = torch.matmul(free_support_function_matrix,coefficients_matrix)
        phi1 = phis[:,0].reshape(len(self.x_train),1)
        phi2 = phis[:,1].reshape(len(self.x_train),1)
        phi3 = phis[:,2].reshape(len(self.x_train),1)
        phi4 = phis[:,3].reshape(len(self.x_train),1)
        # phi5 = phis[:,4].reshape(len(self.x_train),1)
        d_phis = torch.matmul(d_free_support_function_matrix,coefficients_matrix)
        d_phi1 = d_phis[:,0].reshape(len(self.x_train),1)
        d_phi2 = d_phis[:,1].reshape(len(self.x_train),1)
        d_phi3 = d_phis[:,2].reshape(len(self.x_train),1)
        d_phi4 = d_phis[:,3].reshape(len(self.x_train),1)
        # d_phi5 = d_phis[:,4].reshape(len(self.x_train),1)
        
        
        

        phi1_h_init = torch.matmul(-phi1,init_h)
        phi1_x_init = phi1*init_x
        phi1_y_init = phi1*init_y
        phi1_theta_init = phi1*init_theta
        
        phi2_h_predf = torch.matmul(-phi2,final_pred_h)
        phi2_x_predf = phi2*final_pred_x
        phi2_y_predf = phi2*final_pred_y
        phi2_theta_predf = phi2*final_pred_theta

        phi3_dh_init = torch.matmul(-phi3,init_dh)
        phi3_dx_init = phi3*init_dx
        phi3_dy_init = phi3*init_dy
        phi3_dtheta_init = phi3*init_dtheta

        phi4_dh_predf = torch.matmul(-phi4,final_pred_dh)
        phi4_dx_predf = phi4*final_pred_dx
        phi4_dy_predf = phi4*final_pred_dy
        phi4_dtheta_predf = phi4*final_pred_dtheta

        # phi5_dh_final = torch.matmul(-phi5,final_dh)
        # phi5_dx_final = phi5*final_total_dx
        # phi5_dy_final = phi5*final_total_dy
        # phi5_dtheta_final = phi5*final_total_dtheta

        dphi1_h_init = torch.matmul(-d_phi1,init_h)
        dphi1_x_init = d_phi1*init_x
        dphi1_y_init = d_phi1*init_y
        dphi1_theta_init = d_phi1*init_theta
        
        dphi2_h_predf = torch.matmul(-d_phi2,final_pred_h)
        dphi2_x_predf = d_phi2*final_pred_x
        dphi2_y_predf = d_phi2*final_pred_y
        dphi2_theta_predf = d_phi2*final_pred_theta

        dphi3_dh_init = torch.matmul(-d_phi3,init_dh)
        dphi3_dx_init = d_phi3*init_dx
        dphi3_dy_init = d_phi3*init_dy
        dphi3_dtheta_init = d_phi3*init_dtheta

        dphi4_dh_predf = torch.matmul(-d_phi4,final_pred_dh)
        dphi4_dx_predf = d_phi4*final_pred_dx
        dphi4_dy_predf = d_phi4*final_pred_dy
        dphi4_dtheta_predf = d_phi4*final_pred_dtheta

        # dphi5_dh_final = torch.matmul(-d_phi5,final_dh)
        # dphi5_dx_final = d_phi5*final_total_dx
        # dphi5_dy_final = d_phi5*final_total_dy
        # dphi5_dtheta_final = d_phi5*final_total_dtheta


        
        # hx = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf).add(phi5_dh_final),bx).reshape(self.x_train.shape)\
        # .add(phi1_x_init).add(phi2_x_predf).add(phi3_dx_init/self.c).add(phi4_dx_predf/self.c).add(phi5_dx_final/self.c))[:,0]
           
        # dhx = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf).add(dphi5_dh_final),bx).reshape(self.x_train.shape)\
        # .add(dphi1_x_init).add(dphi2_x_predf).add(dphi3_dx_init/self.c).add(dphi4_dx_predf/self.c).add(dphi5_dx_final/self.c))[:,0]
       
        # hy = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf).add(phi5_dh_final),by).reshape(self.x_train.shape)\
        # .add(phi1_y_init).add(phi2_y_predf).add(phi3_dy_init/self.c).add(phi4_dy_predf/self.c).add(phi5_dy_final/self.c))[:,0]
        
        # dhy = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf).add(dphi5_dh_final),by).reshape(self.x_train.shape)\
        # .add(dphi1_y_init).add(dphi2_y_predf).add(dphi3_dy_init/self.c).add(dphi4_dy_predf/self.c).add(dphi5_dy_final/self.c))[:,0]

        # htheta = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf).add(phi5_dh_final),btheta).reshape(self.x_train.shape)\
        # .add(phi1_theta_init).add(phi2_theta_predf).add(phi3_dtheta_init/self.c).add(phi4_dtheta_predf/self.c).add(phi5_dtheta_final/self.c))[:,0]
     
        # dhtheta = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf).add(dphi5_dh_final),btheta).reshape(self.x_train.shape)\
        # .add(dphi1_theta_init).add(dphi2_theta_predf).add(dphi3_dtheta_init/self.c).add(dphi4_dtheta_predf/self.c).add(dphi5_dtheta_final/self.c))[:,0]

        hx = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf),bx).reshape(self.x_train.shape)\
        .add(phi1_x_init).add(phi2_x_predf).add(phi3_dx_init/self.c).add(phi4_dx_predf/self.c))[:,0]
           
        dhx = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf),bx).reshape(self.x_train.shape)\
        .add(dphi1_x_init).add(dphi2_x_predf).add(dphi3_dx_init/self.c).add(dphi4_dx_predf/self.c))[:,0]
       
        hy = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf),by).reshape(self.x_train.shape)\
        .add(phi1_y_init).add(phi2_y_predf).add(phi3_dy_init/self.c).add(phi4_dy_predf/self.c))[:,0]
        
        dhy = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf),by).reshape(self.x_train.shape)\
        .add(dphi1_y_init).add(dphi2_y_predf).add(dphi3_dy_init/self.c).add(dphi4_dy_predf/self.c))[:,0]

        htheta = (torch.matmul(h.add(phi1_h_init).add(phi2_h_predf).add(phi3_dh_init).add(phi4_dh_predf),btheta).reshape(self.x_train.shape)\
        .add(phi1_theta_init).add(phi2_theta_predf).add(phi3_dtheta_init/self.c).add(phi4_dtheta_predf/self.c))[:,0]
     
        dhtheta = (self.c*torch.matmul(dh.add(dphi1_h_init).add(dphi2_h_predf).add(dphi3_dh_init).add(dphi4_dh_predf),btheta).reshape(self.x_train.shape)\
        .add(dphi1_theta_init).add(dphi2_theta_predf).add(dphi3_dtheta_init/self.c).add(dphi4_dtheta_predf/self.c))[:,0]

        # plt.figure()
        # plt.plot(hx.cpu().detach().numpy(),hy.cpu().detach().numpy())
        # plt.plot(self.y_train[:,0],self.y_train[:,1])
        # plt.show()

        # plt.figure()
        # plt.plot(hx.cpu().detach().numpy())
        # plt.plot(self.y_train[:,0])
        # plt.scatter(0,init_x)
        # plt.scatter(len(self.x_train_pred)-1,final_pred_x)
        # plt.show()
        # plt.figure()
        # plt.plot(dhx.cpu().detach().numpy())
        # plt.plot(self.speed_x)
        # plt.scatter(0,init_dx)
        # plt.scatter(len(self.x_train_pred)-1,final_pred_dx)
        # plt.show()
        # plt.figure()
        # plt.plot(hy.cpu().detach().numpy())
        # plt.plot(self.y_train[:,1])
        # plt.scatter(0,init_y)
        # plt.scatter(len(self.y_train_pred)-1,final_pred_y)
        # plt.show()
        # plt.figure()
        # plt.plot(dhy.cpu().detach().numpy())
        # plt.plot(self.speed_y)
        # plt.scatter(0,init_dy)
        # plt.scatter(len(self.x_train_pred)-1,final_pred_dy)
        # plt.show()
        # plt.figure()
        # plt.plot(htheta.cpu().detach().numpy())
        # plt.plot(self.y_train[:,2])
        # plt.scatter(0,init_theta)
        # plt.scatter(len(self.x_train_pred)-1,final_pred_theta)
        # plt.show()
        # plt.figure()
        # plt.plot(dhtheta.cpu().detach().numpy())
        # plt.plot(self.heading_ratio)
        # plt.scatter(0,init_dtheta)
        # plt.scatter(len(self.x_train_pred)-1,final_pred_dtheta)
        # plt.show()
        
        return torch.vstack((hx,hy,htheta))