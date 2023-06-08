import torch
from torch.autograd.functional import jacobian
import numpy as np
import pandas as pd
import datetime
import torch.nn as nn

class PIELM:

    def __init__(self,n_nodes,input_size,output_size,length,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh"):
        # if len(functions)==output_size:
        #     raise ValueError("gotta match number of states predicted and diferential equations")
        # self.functions = functions
        self.length= length
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nodes = n_nodes
        self.W = (torch.randn(size=(n_nodes,1),dtype=torch.float)*(high_w-low_w)+low_w)
        self.b = (torch.randn(size=(n_nodes,1),dtype=torch.float)*(high_b-low_b)+low_b)
        
        self.betas = torch.ones(size=(output_size*n_nodes,),requires_grad=True,dtype=torch.float)
        

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
      
        #self.W.to(device)
        #self.b.to(device)
        
        #self.betas.to(device)
        print(self.betas.is_cuda)
        print("number of samples:",len(self.x_train))
        while count < n_iterations:
            
            with torch.no_grad():
                
                jac = jacobian(self.predict_jacobian,self.betas)
                print(jac.shape)
                loss = self.predict_loss(self.x_train,self.y_train_pred,self.x_train_pred)
                print(loss.shape)
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

        l_pred_x = self.y_train_pred[:,0]-hx
        l_pred_y = self.y_train_pred[:,1]-hy
        l_pred_theta = self.y_train_pred[:,2]-htheta
        #l_pred_delta = self.y_train[:,3]-torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*3:4*self.nodes])
        
        dhx = self.c*torch.matmul(self.get_dh(self.x_train),betas[0:self.nodes])
        dhy = self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes:2*self.nodes])
        dhtheta = self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes*2:3*self.nodes])


        l_x = dhx-(dhx**2+dhy**2)**(1/2)*torch.cos(htheta_full+self.slip_angle)
        l_y = dhy-(dhx**2+dhy**2)**(1/2)*torch.sin(htheta_full+self.slip_angle)
        l_theta = dhtheta-(dhx**2+dhy**2)**(1/2)*torch.tan(self.steering_angle)*torch.cos(self.slip_angle)/self.l
        #l_delta = torch.matmul(self.get_dh(self.x_train),betas[self.nodes*3:4*self.nodes])-self.rho
        l_pred_dhx = self.speed_x- dhx
        l_pred_dhy = self.speed_y- dhy 

        #loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        # loss= torch.hstack((l_pred_x,l_pred_y,l_x,l_y))
        loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                            self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,\
                            (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))
   
        return loss
            
    def predict_loss(self,x,y,x_pred):
       

        hx = torch.matmul(self.get_h(x_pred),self.betas[0:self.nodes])
        hy = torch.matmul(self.get_h(x_pred),self.betas[self.nodes:2*self.nodes])
        htheta = torch.matmul(self.get_h(x_pred),self.betas[self.nodes*2:3*self.nodes])
        htheta_full = torch.matmul(self.get_h(self.x_train),self.betas[self.nodes*2:3*self.nodes])

        l_pred_x = y[:,0]-hx
        l_pred_y = y[:,1]-hy
        l_pred_theta = y[:,2]-htheta
        
        #l_pred_delta = self.y_train[:,3]-torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*3:4*self.nodes])
        
        dhx = self.c*torch.matmul(self.get_dh(x),self.betas[0:self.nodes])
        dhy = self.c*torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])
        dhtheta = self.c*torch.matmul(self.get_dh(x),self.betas[self.nodes*2:3*self.nodes])
        
      
        #l_pred_dhy = 

        l_x = dhx-(dhx**2+dhy**2)**(1/2)*torch.cos(htheta_full+self.slip_angle)
        l_y = dhy-(dhx**2+dhy**2)**(1/2)*torch.sin(htheta_full+self.slip_angle)
        l_theta = dhtheta-(dhx**2+dhy**2)**(1/2)*torch.tan(self.steering_angle)*torch.cos(self.slip_angle)/self.l
        #l_delta = torch.matmul(self.get_dh(self.x_train),betas[self.nodes*3:4*self.nodes])-self.rho
        l_pred_dhx = self.speed_x- dhx
        l_pred_dhy = self.speed_y- dhy 
        #loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        # loss= torch.hstack((l_pred_x,l_pred_y,l_x,l_y))
        loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                            self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,\
                            (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))
   
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
    def __init__(self,n_nodes,input_size,output_size,length,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh"):
        super().__init__(n_nodes,input_size,output_size,length,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh")

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
        
        hx = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[0:self.nodes])\
            + self.y_train[0,0]
        hy = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes:self.nodes*2])\
              + self.y_train[0,1]
        htheta = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*2:self.nodes*3])\
              + self.y_train[0,2]
        htheta_full = torch.matmul(torch.add(self.get_h(self.x_train),-self.get_h(self.x_train_pred[0])),betas[self.nodes*2:self.nodes*3])\
              + self.y_train[0,2]
        # hdelta = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*3:self.nodes*4]) + self.y_train[0,3]
        l_pred_x = self.y_train_pred[:,0]-hx
        l_pred_y = self.y_train_pred[:,1]-hy
        l_pred_theta = self.y_train_pred[:,2]-htheta
        # l_pred_delta = self.y_train[:,3]-hdelta
        
        
        dhx =  self.c*torch.matmul(self.get_dh(self.x_train),betas[0:self.nodes])
        dhy = self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes:2*self.nodes])
        dhtheta =  self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes*2:3*self.nodes])
        # dhdelta = torch.matmul(self.get_dh(self.x_train),betas[self.nodes*3:4*self.nodes])

        l_x = dhx-(((dhx)**2+ (dhy)**2)**(1/2)*torch.cos(htheta_full+self.slip_angle))
        l_y = dhy-(((dhx)**2+ (dhy)**2)**(1/2)*torch.sin(htheta_full+self.slip_angle)) 
        l_theta = dhtheta - (((dhx)**2+ (dhy)**2)**(1/2))*torch.tan(self.steering_angle)*torch.cos(self.slip_angle)/self.l
        # l_delta = dhdelta-self.rho
        l_pred_dhx = self.speed_x - dhx
        l_pred_dhy = self.speed_y - dhy
        l_pred_dhtheta = self.heading_ratio - dhtheta 
        #loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        # loss= torch.hstack((l_pred_x,l_pred_y,l_x,l_y))
        # loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                            # (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))
        # loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                            # self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,self.lambda_*l_pred_dhtheta,\
                            # (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))  
        loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                            self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,\
                            (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))  

        print(loss.shape)
        return loss
            
    def predict_loss(self,x,y,x_pred):
         
        hx = torch.matmul(torch.add(self.get_h(x_pred),-self.get_h(x_pred[0])),self.betas[0:self.nodes])+ self.y_train[0,0]
        hy = torch.matmul(torch.add(self.get_h(x_pred),-self.get_h(x_pred[0])),self.betas[self.nodes:self.nodes*2]) + self.y_train[0,1]
        htheta = torch.matmul(torch.add(self.get_h(x_pred),-self.get_h(x_pred[0])),self.betas[self.nodes*2:self.nodes*3]) + self.y_train[0,2]
        htheta_full = torch.matmul(torch.add(self.get_h(x),-self.get_h(x_pred[0])),self.betas[self.nodes*2:self.nodes*3]) + self.y_train[0,2]
        #hdelta = torch.matmul(torch.add(self.get_h(x_pred),-self.get_h(x_pred[0])),self.betas[self.nodes*3:self.nodes*4]) + self.y_train[0,3]
        
        
        l_pred_x = y[:,0]-hx
        l_pred_y = y[:,1]-hy
        l_pred_theta = y[:,2]-htheta
        # l_pred_delta = y[:,3]-hdelta


        dhx = self.c* torch.matmul(self.get_dh(x),self.betas[0:self.nodes])
        dhy = self.c*torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])
        dhtheta = self.c* torch.matmul(self.get_dh(x),self.betas[self.nodes*2:3*self.nodes])
        # dhdelta = torch.matmul(self.get_dh(x),self.betas[self.nodes*3:4*self.nodes])

        l_x = dhx-(((dhx)**2+ (dhy)**2)**(1/2)*torch.cos(self.y_train[:,2]))
        l_y = dhy-(((dhx)**2+ (dhy)**2)**(1/2)*torch.sin(self.y_train[:,2])) 
        l_theta = dhtheta - (((dhx)**2+ (dhy)**2)**(1/2))*torch.tan(self.steering_angle)*torch.cos(self.slip_angle)/self.l
        # l_delta = dhdelta-self.rho
        
        l_pred_dhx = self.speed_x- dhx
        l_pred_dhy = self.speed_y- dhy
        l_pred_dhtheta = self.heading_ratio - dhtheta 
        #loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        # loss= torch.hstack((l_pred_x,l_pred_y,l_x,l_y))
        # loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                            # (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta)) 
        # loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                            # self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,self.lambda_*l_pred_dhtheta,\
                            # (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))
        loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                            self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,\
                            (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))  
  
        return loss




        # x0 = -torch.matmul(self.get_h(x_pred),self.betas[0:self.nodes])[0]+y[0,0]
        # y0 = -torch.matmul(self.get_h(x_pred),self.betas[self.nodes:2*self.nodes])[0]+y[0,1]
        # theta0 = -torch.matmul(self.get_h(x_pred),self.betas[self.nodes*2:3*self.nodes])[0]+y[0,2]
        # delta0 =-torch.matmul(self.get_h(x_pred),self.betas[self.nodes*3:4*self.nodes])[0]+y[0,3]

        # hx = torch.matmul(self.get_h(x_pred),self.betas[0:self.nodes]) + x0
        # hy = torch.matmul(self.get_h(x_pred),self.betas[self.nodes:2*self.nodes]) +y0
        # htheta = torch.matmul(self.get_h(x_pred),self.betas[self.nodes*2:3*self.nodes]) +theta0
        # hdelta = torch.matmul(self.get_h(x_pred),self.betas[self.nodes*3:4*self.nodes]) +delta0

        # l_pred_x = y[:,0]-hx
        # l_pred_y = y[:,1]-hy
        # l_pred_theta = y[:,2]-htheta
        # l_pred_delta = y[:,3]-hdelta
        
        # l_x = torch.matmul(self.get_dh(x),self.betas[0:self.nodes])-\
        # (torch.matmul(self.get_dh(x),self.betas[0:self.nodes])**2+\
        # torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])**2)**(1/2)\
        # *torch.cos(torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes])+theta0)
        
        # l_y = torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])-\
        # (torch.matmul(self.get_dh(x),self.betas[0:self.nodes])**2+\
        # torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])**2)**(1/2)\
        # *torch.sin(torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes])+theta0) 
        
        # l_theta = torch.matmul(self.get_dh(x),self.betas[self.nodes*2:3*self.nodes])-\
        # (torch.matmul(self.get_dh(x),self.betas[0:self.nodes])**2\
        # +torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])**2)**(1/2)\
        # *torch.tan(torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes])+delta0)/self.l
        
        # l_delta = torch.matmul(self.get_dh(x),self.betas[self.nodes*3:4*self.nodes])-self.rho
        # loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        
        # return loss

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
        print(min(x))
        print(max(x))
        hx = torch.matmul(torch.add(self.get_h(x),-self.get_h(x[0])),self.betas[0:self.nodes])+ self.y_train[0,0]
        hy = torch.matmul(torch.add(self.get_h(x),-self.get_h(x[0])),self.betas[self.nodes:self.nodes*2]) + self.y_train[0,1]
        htheta = torch.matmul(torch.add(self.get_h(x),-self.get_h(x[0])),self.betas[self.nodes*2:self.nodes*3]) + self.y_train[0,2]
        delta_pred = torch.matmul(self.get_h(x),self.betas[self.nodes*3:4*self.nodes])  
        
        return torch.vstack((hx,hy,htheta,delta_pred))