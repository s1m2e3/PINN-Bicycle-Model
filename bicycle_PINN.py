import tensorflow as tf
import numpy as np
import pandas as pd
import time


class bicycle_PINN:
    
    def __init__(self,df,layers):
        
        #Predict future states passing initial points and set of controls

        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        self.t = df["timestamp_posix"]
        self.id = df['temporaryId']

        self.x = df['latitude']
        self.y =  df["longitude"]
        self.theta = df["heading"]
        self.delta = df["steering_angle"]
        self.v = df["speed"]
        self.x_t = df["speed"]*np.cos(self.theta)
        self.y_t = df["speed"]*np.sin(self.theta)
        self.x_tt = df["accel"]*np.cos(self.theta)
        self.y_tt = df["accel"]*np.sin(self.theta)
        
        self.theta_t = df["ang_speed"]
        self.delta_t = df["steering_angle_rate"]

        self.L = df["Length"]
        
        self.lambda_1=tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2=tf.Variable([0.0], dtype=tf.float32)
        self.lambda_3=tf.Variable([0.0], dtype=tf.float32)
        self.lambda_4=tf.Variable([0.0], dtype=tf.float32)
        self.lambda_5=tf.Variable([0.0], dtype=tf.float32)
        self.lambda_6=tf.Variable([0.0], dtype=tf.float32)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.t_tf=tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.x_tf=tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf=tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.theta_tf=tf.placeholder(tf.float32, shape=[None, self.theta.shape[1]])
        self.delta_tf=tf.placeholder(tf.float32, shape=[None, self.delta.shape[1]])
        self.x_t_tf = tf.gradients(self.x_tf)
        
        preds,funcs, boundaries = self.net_NS(self.t_tf,self.x_tt,self.y_tt,self.delta_t,self.L)
        real = [self.x,self.y,self.theta,self.delta,self.v]
        
        pred_error = real-preds

        self.loss = tf.reduce_sum(tf.square(pred_error)) + \
                    tf.reduce_sum(tf.square(funcs)) + \
                    tf.reduce_sum(tf.square(boundaries))  
                    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def net_NS(self,t_tf,x_tt_,y_tt_,delta_t,L):

        x_pred,y_pred,theta_pred,delta_pred,v_pred,=self.neural_net(tf.concat([x_tt_,y_tt_,delta_t,t_tf], 1), self.weights, self.biases)
        
        x_t = tf.gradients(x_pred,t_tf)*np.cos(theta_pred)
        y_t = tf.gradients(y_pred,t_tf)*np.sin(theta_pred)
        x_tt = tf.gradients(x_t,t_tf)
        y_tt = tf.gradients(y_t,t_tf)
        theta_t = tf.gradients(theta_pred,t_tf)
        delta_t = tf.gradients(delta_pred,t_tf)

        f_1 = x_t-self.lambda_1*v_pred*np.cos(theta_pred)
        f_2 = y_t-self.lambda_2*v_pred*np.sin(theta_pred)
        f_3 = x_tt - self.lambda_3*x_tt_
        f_4 = y_tt - self.lambda_4*y_tt_
        f_5 = theta_t - self.lambda_5*v_pred*np.tan(delta_pred)/L
        f_6 = delta_t - self.lambda_6*delta_t

        b_1 = self.x[0]-x_pred[0]
        b_2 = self.y[0]-y_pred[0]
        b_3 = self.theta[0]-theta_pred[0]
        b_4 = self.delta[0]-delta_pred[0]
        b_5 = self.v[0]-v_pred[0]
        

        preds = [x_pred,y_pred,theta_pred,delta_pred,v_pred]
        funcs = [f_1,f_2,f_3,f_4,f_5,f_6]
        boundaries = [b_1,b_2,b_3,b_4,b_5]


        return preds,funcs,boundaries



    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def train(self, nIter): 

            tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                    self.u_tf: self.u, self.v_tf: self.v}
            
            start_time = time.time()
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    lambda_1_value = self.sess.run(self.lambda_1)
                    lambda_2_value = self.sess.run(self.lambda_2)
                    print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % 
                        (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
                    start_time = time.time()
                
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss, self.lambda_1, self.lambda_2],
                                    loss_callback = self.callback)
                
        
        def predict(self, x_star, y_star, t_star):
            
            tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}
            
            u_star = self.sess.run(self.u_pred, tf_dict)
            v_star = self.sess.run(self.v_pred, tf_dict)
            p_star = self.sess.run(self.p_pred, tf_dict)
            
            return u_star, v_star, p_star

