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
        self.ratio = 0.9
        
        self.states = df[['x','y','heading']]
        self.control = df[["speed",'steering_angle',"length","timestamp_posix"]]
        self.XTFC(self.control.shape,self.states.shape,self.n_nodes)

        control_train=self.control.loc[:int(len(self.control)*self.ratio)]
        control_test=self.control.loc[int(len(self.control)*self.ratio):]
        states_train=self.states.loc[:int(len(self.states)*self.ratio)]
        states_test=self.states.loc[int(len(self.states)*self.ratio):]

        states_pred,funcs_pred,bounds=self.preds(self.act,control_train,states_train)
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.train(states_pred-states_train,funcs_pred,bounds,opt)
        

    def XTFC(self,control_shape,states_shape,n_nodes):

        LBw = -3
        LBb = -3
        UBw = 3
        UBb = 3
        self.W=tf.random.uniform(minval=LBw,maxval=UBw,shape=(1,n_nodes))
        self.b=tf.random.uniform(minval=LBw,maxval=UBw,shape=(n_nodes,1))
        self.betas = tf.Variable(np.ones((n_nodes,states_shape[1])),dtype='float32')
        
    def preds(self,activation_function,control,states):
        
        states_shape = states.shape
        u0 = np.array(states.loc[0])
        X = np.array(control)
        ub = X.min()
        lb = X.max()
        H = (X-lb)/(ub-lb)
        H =  tf.convert_to_tensor(H[:,3],dtype='float32')
        
        with tf.GradientTape() as g:
            g.watch(H)
            pred =tf.matmul(tf.tanh(tf.transpose(self.b) +tf.matmul(tf.transpose([H]),self.W)),(self.betas))
            dH_dt = g.jacobian(pred, H)
        #find jacobian for each timestamp prediction
        x_t = tf.linalg.diag_part(dH_dt[:,0,:])
        y_t = tf.linalg.diag_part(dH_dt[:,1,:])
        theta_t = tf.linalg.diag_part(dH_dt[:,2,:])
        
        f1 = x_t - (X[:,0]*tf.cos(pred[:,2]))
        f2 = y_t - (X[:,0]*tf.sin(pred[:,2]))
        f3 = theta_t - (X[:,0]*X[:,1]/X[:,2])
        
        b = pred[0,:]-u0
        print(b)
        return pred, [f1,f2,f3],b
    
    def train(self,res,f,b,opt,n_epochs=4000):       
       
        for i in range(n_epochs):
            with tf.GradientTape() as g:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                g.watch(self.betas)
                loss = tf.reduce_mean(tf.square(res)+tf.square(tf.norm(f))+tf.square(tf.norm(b)))
                grads = g.gradient(loss,self.betas)
                print(grads)
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            opt.apply_gradients(zip(grads,self.betas))

                # Log every 200 batches.
            if i % 200 == 0:
                print("Training loss at step %d: %.4f"%(i, float(loss)))