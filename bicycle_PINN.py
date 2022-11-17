import tensorflow as tf
import numpy as np
import pandas as pd
import datetime


class bicycle_PINN:
    
    def __init__(self,df,mod_type,nodes=10):
        
        #Predict future states passing initial points and set of controls

        # Initialize NN
        self.df = df
        self.n_nodes = nodes
        self.act = "tanh"
        self.ratio = 0.9
        
        self.states = df[['x','y','heading']]
        self.control = df[["speed",'steering_angle',"length","timestamp_posix"]]
        self.XTFC(self.control.shape,self.states.shape,self.n_nodes)

        
        control_train=self.control.loc[:int(len(self.control)*self.ratio)]
        control_test=self.control.loc[int(len(self.control)*self.ratio):]
        states_train=self.states.loc[:int(len(self.states)*self.ratio)]
        states_test=self.states.loc[int(len(self.states)*self.ratio):]
        self.train(self.act,control_train,states_train)
        

    def XTFC(self,control_shape,states_shape,n_nodes):

        LBw = -3
        LBb = -3
        UBw = 3
        UBb = 3
        self.W=tf.random.uniform(minval=LBw,maxval=UBw,shape=(1,n_nodes))
        self.b=tf.random.uniform(minval=LBb,maxval=UBb,shape=(n_nodes,1))
        self.betas = tf.Variable(np.ones((n_nodes,states_shape[1])),dtype='float32')
        
    def train(self,activation_function,control,states):
        
        states_shape = states.shape
        u0 = np.array(states.loc[0])
        X = np.array(control)
        now = datetime.datetime.now()
    
        #bound input time from 0 to 1 
        x_max = 1
        x_min = 0
        c = (x_max-x_min) /(X[:,3].max()-X[:,3].min())
        H =  tf.convert_to_tensor(c*X[:,3],dtype='float32')
        Y = np.array(states)
        H_ = tf.tanh(tf.transpose(self.b) +tf.matmul(tf.transpose([H]),self.W))
        pred = tf.matmul(H_,(self.betas))
        print(tf.reduce_sum(tf.math.pow(pred-Y,2)))
        sech_=1-tf.pow(tf.tanh(tf.transpose(self.b) +tf.matmul(tf.transpose([H]),self.W)),2)
        right_mult = tf.math.multiply(sech_,self.W)
        dH_dt_ = right_mult*c
        A = tf.concat([H_,dH_dt_],axis=0)
        xt=(tf.transpose([X[:,0]*tf.cos(pred[:,2])]))
        yt=(tf.transpose([X[:,0]*tf.sin(pred[:,2])]))
        thetat=(tf.transpose([X[:,0]*X[:,1]/X[:,2]]))
        thetat =tf.cast(thetat,tf.float32)
        U =tf.concat([xt,yt],axis=1)
        U = tf.concat([U,thetat],axis=1)
        B = tf.concat([Y,U],axis=0)
        self.betas = tf.linalg.lstsq(A,B,fast=False)
        after = datetime.datetime.now()
        print(after-now)
        new_preds = tf.matmul(H_,(self.betas))
        print(tf.reduce_sum(tf.math.pow(new_preds-Y,2)))

         # with tf.GradientTape(persistent=True) as g:
        #     g.watch(H)
        #     g.watch(self.betas)
        #     dH_dt = tf.matmul(right_mult,self.betas)
        #     x_t = dH_dt[:,0]
        #     y_t = dH_dt[:,1]
        #     theta_t = dH_dt[:,2]
        #     f1 = x_t - (X[:,0]*tf.cos(pred[:,2]))
        #     f2 = y_t - (X[:,0]*tf.sin(pred[:,2]))
        #     f3 = theta_t - (X[:,0]*X[:,1]/X[:,2])
        #     b = pred[0,:]-u0
        #     loss = tf.reduce_mean(tf.square(pred-Y)+tf.square(tf.norm([f1+f2+f3]))+tf.square(tf.norm(b)))
                
        