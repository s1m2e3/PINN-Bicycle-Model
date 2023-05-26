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


def get_curvature(data):
    data["slip_angle"] = 0
    data["curvature_radius"]=0

    ## compute bisector
    for id in data["temporaryId"].unique():
        sub_data = data[data["temporaryId"]==id]
        for sub in sub_data["sub_group"].unique():
            subdf = sub_data[sub_data["sub_group"]==sub]
            for  index,row   in subdf.iterrows():
                if index != subdf.index[-1]:
                    delta_time = subdf["timestamp_posix"].loc[index+1]-subdf["timestamp_posix"].loc[index]
                    delta_x = subdf["x"].loc[index+1]-subdf["x"].loc[index]
                    delta_y = subdf["y"].loc[index+1]-subdf["y"].loc[index]
                    subdf["speed_x"].loc[index]=delta_x/delta_time
                    subdf["speed_y"].loc[index]=delta_y/delta_time
                    
                    subdf["slip_angle"].loc[index]=np.arctan(subdf["speed_y"].loc[index]/subdf["speed_x"].loc[index])
                    subdf["steering_angle"].loc[index]=np.arctan(subdf["length"].loc[index]*np.tan(subdf["slip_angle"].loc[index]))
                    

            indices = subdf.index
            data.loc[indices]=subdf
    return data


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
    df = df[df["sub_group"]==df["sub_group"].loc[1]].reset_index(drop=True)
    df["heading"] = df["heading"]*np.pi/180
    test_df = df
    n_iterations = int(1e3)
    

    ##compute curvature radius
    df = get_curvature(df)

    ## neural networks with only time:

    n_nodes = 512
    input_sequence_length = 50
    output_sequence_length = 10
    ratio = 10
    stop = int(len(test_df)/10*ratio)
    accuracy = 1e-5
    
    layers = 2
    hidden = 10

    ##PIELM - XTFC
    n_nodes = 60
    input_size = 1
    output_size = 4

    pielm_x = np.array((test_df["timestamp_posix"]))
    pielm_y = np.array((test_df[["x","y","heading"]]))
    # l = np.array((test_df["length"]-test_df["length"].min())/(test_df["length"].max()-test_df["length"].min()))
    # rho = np.array((test_df["steering_angle_rate"]-test_df["steering_angle_rate"].min())/\
    #     (test_df["steering_angle_rate"].max()-test_df["steering_angle_rate"].min()))
    l = np.array(test_df['length'])
    rho = np.array(test_df["steering_angle_rate"])
    steering_angle = np.array(test_df["steering_angle"])
    slip_angle = np.array(test_df["slip_angle"])
    speed_x = np.array(test_df["speed_x"])
    speed_y = np.array(test_df["speed_y"])

    pielm_x_train = pielm_x[:stop] 
    pielm_x_test = pielm_x[stop:] 
    pielm_y_train = pielm_y[:stop]
    pielm_y_test = pielm_y[stop:] 
    l_train = l[:stop]
    l_test = l[stop:] 
    rho_train = rho[:stop]
    rho_test = rho[stop:]
    steering_angle_train = steering_angle[:stop]
    slip_angle_train = slip_angle[:stop]
    speed_x= speed_x[:stop]
    speed_y= speed_y[:stop]
    

    # pielm= PIELM(n_nodes,input_size,output_size,low_w=-1,high_w=1,low_b=-1,high_b=1,activation_function="tanh",length=10)
    # pielm.train(accuracy, n_iterations,pielm_x_train,pielm_y_train,l_train,rho_train,steering_angle_train,slip_angle_train,0.5)
    # y_pred = pielm.pred(pielm_x_train).cpu().detach().numpy().T
    
    # # plt.figure()
    # # plt.scatter(pielm_y_train[:,0],pielm_y_train[:,1])
    # # plt.scatter(y_pred[:,0],y_pred[:,1])
    # # plt.show()
    # plt.figure()
    # plt.plot(pielm_y_train[:,0])
    # plt.plot(y_pred[:,0])
    # plt.show()
    # plt.figure()
    # plt.plot(pielm_y_train[:,1])
    # plt.plot(y_pred[:,1])
    # plt.show()
    # plt.figure()
    # plt.plot(pielm_y_train[:,2])
    # plt.plot(y_pred[:,2])
    # plt.show()
    # plt.figure()
    
    y_pred={}
    for i in [1]:
        y_pred[i]={}
        for j in [1]:
            lambda_ = 1-((j+1)/100)
            length= 10*i
            xtfc= XTFC(n_nodes,input_size,output_size,length=length,low_w=-1,high_w=1,low_b=-1,high_b=1,activation_function="tanh")
            xtfc.train(accuracy, n_iterations,pielm_x_train,pielm_y_train,l_train,rho_train,steering_angle_train,slip_angle_train,speed_x,speed_y,lambda_)
            y_pred[i][j] = xtfc.pred(pielm_x_train).cpu().detach().numpy().T
    plt.figure()
    plt.plot(pielm_y_train[:,0])
    plt.plot(y_pred[1][1][:,0])
    #plt.axvline(x=len(pielm_x_train)-length)
    plt.show()
    plt.figure()
    plt.plot(pielm_y_train[:,1])
    plt.plot(y_pred[1][1][:,1])
    #plt.axvline(x=len(pielm_x_train)-length)
    plt.show()
    plt.figure()
    plt.plot(pielm_y_train[:,2])
    plt.plot(y_pred[1][1][:,2])
    #plt.axvline(x=len(pielm_x_train)-length)
    plt.show()
    
    # for j in range[0]:
    #     plt.figure()
    #     plt.scatter(pielm_y_train[:,0],pielm_y_train[:,1])
    #     for i in y_pred:
    #         plt.scatter(y_pred[i][j][:,0],y_pred[i][j][:,1])
    #         #plt.axvline(x=len(pielm_x_train-10))
    #     plt.legend(["ground_truth","1_sec_ahead","3_sec_ahead","5_sec_ahead"])
    #     plt.title("x and y coordinates alignment")
    #     plt.savefig("alignment_pure_data.png" if j ==0 else "alignment_phys_plus_data.png")

    #     plt.figure()
    #     plt.plot(pielm_y_train[:,0])
    #     for i in y_pred:
    #         plt.plot(y_pred[i][j][:,0])
    #         plt.axvline(x=len(pielm_x_train)-(10*(i+1)),c="black")
    #     plt.legend(["ground_truth","1_sec_ahead","1_sec_cut","3_sec_ahead","3_sec_cut","5_sec_ahead","5_sec_cut"])
    #     plt.title("x coordinates over time")
    #     plt.savefig("x_coordinates_pure_data.png" if j ==0 else "x_coordinates_phys_plus_data.png")

    #     plt.figure()
    #     plt.plot(pielm_y_train[:,1])
    #     for i in y_pred:
    #         plt.plot(y_pred[i][j][:,1])
    #         plt.axvline(x=len(pielm_x_train)-(10*(i+1)),c="black")
    #     #plt.axvline(x=len(pielm_x_train-10))
    #     plt.legend(["ground_truth","1_sec_ahead","1_sec_cut","3_sec_ahead","3_sec_cut","5_sec_ahead","5_sec_cut"])
    #     plt.title("y coordinates over time")
    #     plt.savefig("y_coordinates_pure_data.png" if j ==0 else "y_coordinates_phys_plus_data.png")

    #     plt.figure()
    #     plt.plot(pielm_y_train[:,2])
    #     for i in y_pred:
    #         plt.plot(y_pred[i][j][:,2])
    #         plt.axvline(x=len(pielm_x_train)-(10*(i+1)),c="black")
    #     #plt.axvline(x=len(pielm_x_train-10))
    #     plt.legend(["ground_truth","1_sec_ahead","1_sec_cut","3_sec_ahead","3_sec_cut","5_sec_ahead","5_sec_cut"])
    #     plt.title("heading over time")
    #     plt.savefig("heading_pure_data.png" if j ==0 else "heading_phys_plus_data.png")


    
        





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
    