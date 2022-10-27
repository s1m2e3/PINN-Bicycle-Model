import tensorflow as tf
import numpy as np
import pandas as pd


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
        self.v_t = df["accel"]
        self.theta_t = df["ang_speed"]
        self.delta_t = df["steering_angle_rate"]
    
        
        self.lambda_1=tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2=tf.Variable([0.0], dtype=tf.float32)
        self.lambda_3=tf.Variable([0.0], dtype=tf.float32)
        self.lambda_4=tf.Variable([0.0], dtype=tf.float32)
        self.lambda_5=tf.Variable([0.0], dtype=tf.float32)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.t_tf=tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.x_tf=tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf=tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.theta_tf=tf.placeholder(tf.float32, shape=[None, self.theta.shape[1]])
        self.delta_tf=tf.placeholder(tf.float32, shape=[None, self.delta.shape[1]])
        
        self.x_pred,self.y_pred,self.theta_pred, self.delta_pred, self.v_pred \
            , self.f_x,self.f_y,self.f_v,self.f_theta,self.f_delta = self.net_NS(self.t_tf,self.x_tf,self.y_tf,\
            self.theta_tf,self.delta_tf)

        self.loss = tf.reduce_sum(tf.square(self.x - self.x_pred)) + \
                    tf.reduce_sum(tf.square(self.y - self.y_pred)) + \
                    tf.reduce_sum(tf.square(self.theta-self.theta_pred)) + \
                    tf.reduce_sum(tf.square(self.delta-self.delta_pred))+ \
                    tf.reduce_sum(tf.square(self.f_x)) + \
                    tf.reduce_sum(tf.square(self.f_y)) + \
                    tf.reduce_sum(tf.square(self.f_v)) + \
                    tf.reduce_sum(tf.square(self.f_theta)) + \
                    tf.reduce_sum(tf.square(self.f_delta)) 
                    
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

    def net_NS(self,t_tf,x_tf,y_tf,theta_tf,delta_tf):

        x_t = tf.gradients(x_tf,t_tf)
        y_t = tf.gradients(y_tf,t_tf)
        x_tt = tf.gradients(x_tt,t_tf)
        y_tt = tf.gradients(y_tt,t_tf)
        theta_t = tf.gradients(theta_tf,t_tf)
        delta_t = tf.gradients(delta_tf,t_tf)

        x_pred = 



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