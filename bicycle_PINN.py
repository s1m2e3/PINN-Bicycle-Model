import torch
import numpy as np
import pandas as pd
import time


class bicycle_PINN:
    
    def __init__(self,df,mod_type,nodes=10):
        
        #Predict future states passing initial points and set of controls

        # Initialize NN
        self.df = df
        self.n_nodes = nodes
        self.act = "tanh"
        self.ratio = 0.7
        
        self.states = df[['x','y','speed_x','speed_y','heading','steering_angle']]
        self.control = df[["accel_x",'accel_y',"steering_angle_rate","length","timestamp_posix"]]
        self.XTFC(self.control.shape,self.states.shape,self.n_nodes)

        control_train=self.control.loc[:int(len(self.control)*self.ratio)]
        control_test=self.control.loc[int(len(self.control)*self.ratio):]
        states_train=self.states.loc[:int(len(self.states)*self.ratio)]
        states_test=self.states.loc[int(len(self.states)*self.ratio):]

        states_pred,funcs_pred,b=self.preds(self.act,control_train,states_train)

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                            method = 'L-BFGS-B', 
                                                            options = {'maxiter': 50000,
                                                                        'maxfun': 50000,
                                                                        'maxcor': 50,
                                                                        'maxls': 50,
                                                                        'ftol' : 1.0 * np.finfo(float).eps})
        self.loss = tf.reduce()
       
    def XTFC(self,control_shape,states_shape,n_nodes):

        LBw = -3
        LBb = -3
        UBw = 3
        UBb = 3

        self.W=torch.tensor(np.random.uniform(low=LBw,high=UBw,size=(n_nodes,control_shape[1]))).float()
        self.b=torch.tensor(np.random.uniform(low=LBb,high=UBb,size=(n_nodes))).float()
        self.betas = torch.tensor(np.random.uniform(size=(n_nodes,states_shape[1])),requires_grad=True).float()
        
    def preds(self,activation_function,control,states):
        
        states_shape = states.shape
        u0 = np.array(states.loc[0])
        X = np.array(control)
        ub = X.min(0)
        lb = X.max(0)
        H =  torch.tensor(X,dtype=torch.float32,requires_grad=True)
        grads = []
        preds = torch.tensor(np.zeros((states_shape)),dtype=torch.float32)
        test = torch.tensor(np.ones(control.shape[1]),requires_grad=True,dtype=torch.float32)
        test_pred = torch.matmul(torch.tanh(torch.add(torch.matmul(self.W,test),self.b)),(self.betas))
        test_pred.backward(gradient=torch.tensor(np.ones((6))))
        print(test.grad)
            
        print(H.grad)
        #for i in range(control.shape[0]):
        #    preds[i] = torch.matmul(torch.tanh(torch.add(torch.matmul(self.W,H[i]),self.b)),(self.betas))
        #print(preds)
        #preds.backward(gradient=torch.tensor(np.ones((171,6))))
        #print(H.grad)
        #print(self.betas.grad)
        #print(preds)            
        #print(grads)
        #preds.backward(torch.tensor(np.ones((171,6))))
        #print(H.grad)

        # x_t = g.gradient(x,H[:,4])
        # y_t = g.gradient(y,H[:,4])
        # x_tt = g.gradient(vx,H[:,4])
        # y_tt = g.gradient(vy,H[:,4])
        # theta_t = g.gradient(theta,H[:,4])
        # delta_t = g.gradient(delta,H[:,4])

        
        # f1 = x_t - preds[:,2]
        # f2 = y_t - preds[:,3]
        # f3 = x_tt - H[:,0]
        # f4 = y_tt - H[:,1]
        # #f5 = theta_t - tf.pow(tf.pow(x_t,2)+tf.pow(y_t,2),2)*tf.tan(preds[:,5])/H[:,3]
        # f6 = delta_t - H[:,2]
        
        # b = preds[0,:]-u0
        
        #return preds, [f1,f2,f3,f4,f5,f6],b
    
       