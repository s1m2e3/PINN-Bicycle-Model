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
        
        if mod_type=="reg":

            self.states = df[['x','y','speed','heading','steering_angle']]
            self.control = df[["accel","steering_angle_rate","length","timestamp_posix"]]
            self.XTFC(self.control.shape,self.states.shape,self.n_nodes)

            control_train=self.control.loc[:int(len(self.control)*self.ratio)]
            control_test=self.control.loc[int(len(self.control)*self.ratio):]
            states_train=self.states.loc[:int(len(self.states)*self.ratio)]
            states_test=self.states.loc[int(len(self.states)*self.ratio):]

            states_pred,funcs_pred=self.preds(mod_type,self.act,self.control_train,self.states.shape)

            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
            
            self.loss = 
            self.train()

        elif mod_type=="lin":     
            self.states = df[['x','y','speed','heading','steering_angle']]
            self.control = df[["accel_x","accel_y","ang_speed","timestamp_posix"]]
            self.XTFC = self.XTFC(self.control.shape,self.states.shape)
            self.loss = self.loss(mod_type)
            self.optimizer = self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

    def XTFC(self,control_shape,states_shape,n_nodes):

        LBw = -3
        LBb = -3
        UBw = 3
        Ubb = 3
        
        self.W = tf.random.uniform((n_nodes,control_shape[1]),minval=LBw,maxval=UBw)
        self.b = tf.random.uniform((n_nodes,1),minval=LBw,maxval=UBw)
        self.betas = [tf.Variable(1., shape=tf.TensorShape(n_nodes,1)) for i in states_shape[1]]

    def preds(self,mod_type,activation_function,control,states_shape):
        
        X = np.array(control)
        ub = X.min(0)
        lb = X.max(0)
        H =  2.0*(X - lb)/(ub - lb) - 1.0
        preds = tf.tensor(shape=(control_shape[0],states_shape[1]))
        for i in range(len(X)):
            prev = tf.tanh(tf.add(tf.matmul(self.W,H[i]),self.b))
            for j in range(len(self.betas)):
                preds[i,j]=tf.matmul(prev,beta[j])

        if mod_type=="reg":
            for i in range(len(X)):
                
                x_t = tf.gradients(preds,preds[:,0])
                y_t = tf.gradients(preds,preds[:,1])
                v_t = tf.gradients(preds,preds[:,2])
                theta_t = tf.gradients(preds,preds[:,3])
                delta_t = tf.gradients(preds,preds[:,4])
                
                f1 = x_t - preds[:,2]*tf.cos(preds[:,3])
                f2 = y_t - preds[:,2]*tf.sin(preds[:,3])
                f3 = v_t - X[:,0]
                f4 = theta_t - preds[:,2]*tf.tan(preds[:,4])/X[:,2]
                f5 = delta_t - X[:,1]

                b = preds[0:] - X[0:]
                
                return preds, [f1,f2,f3,f4,f5],b
        
        elif mod_type=="lin":

            for i in range(len(X)):
                
                x_t = tf.gradients(preds,preds[:,0])
                y_t = tf.gradients(preds,preds[:,1])
                x_tt = tf.gradients(preds,preds[:,2])
                y_tt = tf.gradients(preds,preds[:,3])
                theta_t = tf.gradients(preds,preds[:,4])
                
                f1 = preds[:,0] - X[:,0]
                f2 = y_t - X[:,0]
                f3 = v_t - X[:,0]
                f4 = theta_t - X[:,0]
                f5 = delta_t - X[:,0]

                b = preds[0:] - X[0:]
                
                return preds, [f1,f2,f3,f4,f5],b