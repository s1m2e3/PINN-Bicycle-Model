import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from bicycle_PINN import PIELM
from bicycle_PINN import XTFC
import datetime
import utm
#from process import prep_df_reg
#from predict_ode import *
import matplotlib.pyplot as plt
import seaborn as sns
#from process import * 
from datetime import datetime
from model import *


def conver_to_lstm_data(data,sequence_length):
    data =np.array(data)
    
    new_shape = [data.shape[0]-sequence_length]
    #data_shape = list(data.shape)
    
    new_shape.append(sequence_length)
    new_shape.append(data.shape[1])
    data_shape = tuple(new_shape)
    #print(data_shape)
    new_data = np.zeros(shape=data_shape)
    for i in range(len(data)-sequence_length):
        new_data[i]=data[i:i+sequence_length]
    
    return new_data

def main():
    sns.set()
    df = pd.read_csv("edited.csv")
    df = df[df["temporaryId"]==df["temporaryId"].loc[0]].reset_index(drop=True)
    test_df = df
    n_iterations = int(1e0)
    
    ## neural networks with only time:

    n_nodes = 512
    input_sequence_length = 50
    output_sequence_length = 10
    stop = int(len(test_df)/10*7)
    accuracy = 1e-5
    
    layers = 2
    hidden = 10

    ##PIELM - XTFC
    n_nodes = 16
    input_size = 1
    output_size = 4

    pielm_x = np.array((test_df["timestamp_posix"]-test_df["timestamp_posix"].min())/\
        (test_df["timestamp_posix"].max()-test_df["timestamp_posix"].min()))
    pielm_y = np.array((test_df[["x","y","heading","steering_angle"]]-test_df[["x","y","heading","steering_angle"]].min())/\
        (test_df[["x","y","heading","steering_angle"]].max()-test_df[["x","y","heading","steering_angle"]].min()))
    l = np.array((test_df["length"]-test_df["length"].min())/(test_df["length"].max()-test_df["length"].min()))
    rho = np.array((test_df["steering_angle_rate"]-test_df["steering_angle_rate"].min())/\
        (test_df["steering_angle_rate"].max()-test_df["steering_angle_rate"].min()))

    pielm_x_train = pielm_x[:stop] 
    pielm_x_test = pielm_x[stop:] 
    pielm_y_train = pielm_y[:stop]
    pielm_y_test = pielm_y[stop:] 
    l_train = l[:stop]
    l_test = l[stop:] 
    rho_train = rho[:stop]
    rho_test = rho[stop:]

    #pielm= PIELM(n_nodes,input_size,output_size,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh")
    xtfc= XTFC(n_nodes,input_size,output_size,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh")
    xtfc.train(accuracy, n_iterations,pielm_x_train,pielm_x_train,l_train,rho_train)
    #xtfc.train()

    ##neural networks with previous states:
    
    # x = (test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]]-test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]].min())/(test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]].max()-test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]].min())
    # y = np.array(x)
    # x = np.array(x)
    
    
    # ff_nn = NN(x.shape[1]*input_sequence_length,n_nodes,y.shape[1]*output_sequence_length)
    
    # x_train = conver_to_lstm_data(x[0:stop],input_sequence_length)
    # y_train = conver_to_lstm_data(y[0:stop],output_sequence_length)
    # x_test = conver_to_lstm_data(x[stop:],input_sequence_length)
    # y_test = conver_to_lstm_data(y[stop:],output_sequence_length)
    
    # x_train = x_train[:-output_sequence_length]
    # y_train = y_train[input_sequence_length:]
    
    # x_train = x_train.reshape((len(x_train),x.shape[1]*input_sequence_length))
    # y_train = y_train.reshape((len(y_train),y.shape[1]*output_sequence_length))
    
    # ff_nn.train(n_iterations,x_train,y_train)

    # x_test = x_test[:-output_sequence_length]
    # y_test = y_test[input_sequence_length:]
    # x_test = x_test.reshape((len(x_test),x.shape[1]*input_sequence_length))
    # #y_test = y_test.reshape((len(y_test),y.shape[1]*output_sequence_length))
    # pred_ff_nn = ff_nn.forward(x_test).cpu().detach().numpy().reshape(y_test.shape)
    # with open('prediction_nn.txt', 'w') as outfile:
    #     for slice_2d in pred_ff_nn:
    #         np.savetxt(outfile, slice_2d)
    # with open('test_nn.txt', 'w') as outfile:
    #     for slice_2d in y_test:
    #         np.savetxt(outfile, slice_2d)
    
   
        
    # x = (test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]]-test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]].min())/(test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]].max()-test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]].min())
    # y = np.array(x)
    # x = np.array(x)
    

    # lstm = LSTM(x.shape[1],hidden,layers,y.shape[1],input_sequence_length,output_sequence_length)
    # x_train = conver_to_lstm_data(x[0:stop],input_sequence_length)
    # y_train = conver_to_lstm_data(y[0:stop],output_sequence_length)
    # x_test = conver_to_lstm_data(x[stop:],input_sequence_length) 
    # y_test = conver_to_lstm_data(y[stop:],output_sequence_length) 
    
    # x_train = x_train[:-output_sequence_length]
    # y_train = y_train[input_sequence_length:]
    
    # x_test = x_test[:-output_sequence_length]
    # y_test = y_test[input_sequence_length:]

    # #lstm.train(n_iterations,x_train,y_train)
    # pred_lstm= lstm.forward(x_test)
    # y_pred = pred_lstm.cpu().detach().numpy()

    # with open('input_lstm.txt', 'w') as outfile:
    #     for slice_2d in x_test:
    #         np.savetxt(outfile, slice_2d)


    # # with open('prediction_lstm.txt', 'w') as outfile:
    # #     for slice_2d in y_pred:
    # #         np.savetxt(outfile, slice_2d)
    # with open('test_lstm.txt', 'w') as outfile:
    #     for slice_2d in y_test:
    #         np.savetxt(outfile, slice_2d)
    
    
if __name__=="__main__":
    main()
    