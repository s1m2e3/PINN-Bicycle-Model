import tensorflow as tf
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
        
        self.W = tf.random.uniform((n_nodes,control_shape[1]),minval=LBw,maxval=UBw)
        self.b = tf.random.uniform((n_nodes,1),minval=LBb,maxval=UBb)
        self.betas = [tf.Variable(1., shape=tf.TensorShape(n_nodes,1)) for i in states_shape[1]]
        #self.lambdas = [(tf.Variable(1.),tf.Variable(1)) for i in states_shape[1]]

    def preds(self,activation_function,control,states):
        
        states_shape = states.shape
        u0 = np.array(states.loc[0])
        X = np.array(control)
        ub = X.min(0)
        lb = X.max(0)
        H =  2.0*(X - lb)/(ub - lb) - 1.0
        preds = tf.tensor(shape=(control.shape[0],states_shape[1]))
        for i in range(len(X)):
            if activation_function =="tanh":
                prev = tf.tanh(tf.add(tf.matmul(self.W,H[i]),self.b))
                for j in range(len(self.betas)):
                    preds[i,j]=tf.matmul(prev,self.betas[j])

    
        
        x_t = tf.gradients(preds[:,0],X[:,4])
        y_t = tf.gradients(preds[:,1],X[:,4])
        x_tt = tf.gradients(preds[:,2],X[:,4])
        y_tt = tf.gradients(preds[:,3],X[:,4])
        theta_t = tf.gradients(preds[:,4],X[:,4])
        delta_t = tf.gradients(preds[:,5],X[:,4])
        
        f1 = x_t - preds[:,2]
        f2 = y_t - preds[:,3]
        f3 = x_tt - X[:,0]
        f4 = y_tt - X[:,1]
        f5 = theta_t - tf.pow(tf.pow(x_t,2)+tf.pow(y_t,2),2)*tf.tan(preds[:,5])/X[:,3]
        f6 = delta_t - X[:,2]
        
        b = preds[0,:]-u0
        
        return preds, [f1,f2,f3,f4,f5,f6],b
    
       