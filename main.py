import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from bicycle_PINN import PIELM
from bicycle_PINN import XTFC
import datetime
import utm
from process import prep_df_reg
from predict_ode import *
import matplotlib.pyplot as plt
import seaborn as sns
from process import * 
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
    # df = prep_df_reg(df)
    # df.to_csv("edited.csv")
    df = df[df["temporaryId"]==df["temporaryId"].loc[0] ].reset_index(drop=True)
    #test_df=df[df["sub_group"]=='inbound1'].reset_index(drop=True)
    test_df = df
    # test_df["time"]=pd.to_datetime(test_df["timestamp_posix"],unit="s")
    # test_df=test_df.set_index("time")
    # test_df=test_df.resample('1L').mean()
    # test_df = test_df.interpolate(method='linear', limit_direction='forward', axis=0)
    
    #states_reg = test_ode_reg(test_df)
    #compare_states = np.array(test_df[["x","y","heading","timestamp_posix"]])
    l = test_df['length']
    rho = test_df["steering_angle_rate"]
    x = (test_df["timestamp_posix"]-test_df["timestamp_posix"][0])/(test_df["timestamp_posix"][len(test_df)-1]-test_df["timestamp_posix"][0])
    y =(test_df[["x","y","heading","steering_angle"]]-test_df[["x","y","heading","steering_angle"]].min())/(test_df[["x","y","heading","steering_angle"]].max()-test_df[["x","y","heading","steering_angle"]].min())
    #y= test_df[["x","y","heading","steering_angle"]]
    # plt.figure()
    # plt.scatter(y["x"]-y["x"][0],y["y"]-y["y"][0])
    # plt.show()
    # plt.figure()
    # plt.plot(y["heading"])
    # plt.show()
    # plt.figure()
    # plt.plot(y["steering_angle"])
    # plt.show()
    accuracy = 1e-5
    n_iterations = int(1e5)
    n_nodes = 20
    # pielm = XTFC(n_nodes=n_nodes,input_size= x.shape[0],output_size=y.shape[1])
    x = x.iloc[0:20]
    y = y.iloc[0:10]
    l = l.iloc[0:20]
    rho = rho.iloc[0:20]
    #pielm.train(accuracy,n_iterations,x,y,l,rho)    
    #y_pred = pd.DataFrame(pielm.pred(x).detach().numpy()).T
    
    #y_pred.columns = ["x","y","heading","steering_angle"]
    # plt.figure()
    # plt.scatter(y["x"]-y["x"][0],y["y"]-y["y"][0])
    # plt.scatter(y_pred["x"]-y_pred["x"][0],y_pred["y"]-y_pred["y"][0])
    # plt.show()
    # plt.figure()
    # plt.plot(y["heading"])
    # plt.plot(y_pred["heading"])
    # plt.show()
    # plt.figure()
    # plt.plot(y["steering_angle"])
    # plt.plot(y_pred["steering_angle"])
    # plt.show()



    ## neural networks with only time:
    n_nodes = 32
    input_sequence_length = 50
    output_sequence_length = 10
    #x = np.array((test_df["timestamp_posix"]-test_df["timestamp_posix"][0])/(test_df["timestamp_posix"][len(test_df)-1]-test_df["timestamp_posix"][0]))
    #y = np.array((test_df[["x","y","heading","steering_angle"]]-test_df[["x","y","heading","steering_angle"]].min())/(test_df[["x","y","heading","steering_angle"]].max()-test_df[["x","y","heading","steering_angle"]].min()))
    #x =np.array(test_df["timestamp_posix"])
    #y =np.array(test_df[["x","y","heading","steering_angle"]])
    stop = int(len(test_df)/10*7)
    #x = np.reshape(x,(len(x),1))
    
    # ff_nn = NN(x.shape[1],n_nodes,y.shape[1])
    # ff_nn.train(n_iterations,x[0:stop],y[0:stop,])
    # pred_ff_nn = ff_nn.forward(x[stop:])
    
    
    #for batched lstm remember dimensions N*L*Hin, where N is the number of batches, L is the sequence length and Hin is the input dimension
    #the dimension of the output is the number of L* Hidden where L is the sequence length and H is the hidden size
    
    layers = 2
    hidden = 20
    batch_size = 100
    # lstm = LSTM(x.shape[1],hidden,layers,y.shape[1],input_sequence_length,output_sequence_length)
    # x_train = conver_to_lstm_data(x[0:stop],input_sequence_length)
    # y_train = conver_to_lstm_data(y[0:stop],output_sequence_length)
    # lstm.train(n_iterations,x_train,y_train)
    # pred_lstm= lstm.predict(y.loc[stop:])

    # ##neural newtworks with controlers only:
    # x = np.array((test_df[["timestamp_posix","speed","steering_angle_rate"]]-test_df[["timestamp_posix","speed","steering_angle_rate"]].min())/(test_df[["timestamp_posix","speed","steering_angle_rate"]].max()-test_df[["timestamp_posix","speed","steering_angle_rate"]].min()))
    # y = np.array((test_df[["x","y","heading","steering_angle"]]-test_df[["x","y","heading","steering_angle"]].min())/(test_df[["x","y","heading","steering_angle"]].max()-test_df[["x","y","heading","steering_angle"]].min()))
    # ff_nn = NN(x.shape[1],n_nodes,y.shape[1])
    # ff_nn.train(n_iterations,x[0:stop],y[0:stop])
    # pred_ff_nn = ff_nn.predict(x[stop:])

    
    # lstm = LSTM(x.shape[1],hidden,layers,y.shape[1],input_sequence_length,output_sequence_length)
    # x_train = conver_to_lstm_data(x[0:stop],input_sequence_length)
    # y_train = conver_to_lstm_data(y[0:stop],output_sequence_length)
    # lstm.train(n_iterations,x_train,y_train)
    # pred_lstm= lstm.predict(y[stop:])
    

    ##neural networks with previous states:
    #x = (test_df[["timestamp_posix","x","y","heading","steering_angle","speed","steering_angle_rate"]]-test_df[["timestamp_posix","x","y","heading","steering_angle","speed","steering_angle_rate"]].min())/(test_df[["timestamp_posix","x","y","heading","steering_angle","speed","steering_angle_rate"]].max()-test_df[["timestamp_posix","x","y","heading","steering_angle","speed","steering_angle_rate"]].min())
    x = (test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]]-test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]].min())/(test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]].max()-test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]].min())
    x = test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]]
    y = np.array(x.iloc[1:])
    x = np.array(x.iloc[:-2])
    
    ff_nn = NN(x.shape[1],n_nodes,y.shape[1])
    ff_nn.train(n_iterations,x[0:stop],y[0:stop])
    pred_ff_nn = ff_nn.forward(x[stop:])

    y_pred = pd.DataFrame(pred_ff_nn.cpu().detach().numpy())
    
    y_pred.columns = ["x","y","heading","steering_angle","speed","steering_angle_rate"]
    y_pred.to_csv("prediction_nn.csv")
    test_df.to_csv("chosen_data.csv")

    # plt.figure()
    # plt.scatter(y["x"]-y["x"][0],y["y"]-y["y"][0])
    # plt.scatter(y_pred["x"]-y_pred["x"][0],y_pred["y"]-y_pred["y"][0])
    # plt.show()
    # plt.figure()
    # plt.plot(y["heading"])
    # plt.plot(y_pred["heading"])
    # plt.show()
    # plt.figure()
    # plt.plot(y["steering_angle"])
    # plt.plot(y_pred["steering_angle"])
    # plt.show()

    # x = np.array((test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]]-test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]].min())/(test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]].max()-test_df[["x","y","heading","steering_angle","speed","steering_angle_rate"]].min()))
    x = np.array(x)
    y = np.array(x)

    
    lstm = LSTM(x.shape[1],hidden,layers,y.shape[1],input_sequence_length,output_sequence_length)
    x_train = conver_to_lstm_data(x[0:stop],input_sequence_length)
    y_train = conver_to_lstm_data(y[0:stop],output_sequence_length)
    x_test = conver_to_lstm_data(x[stop:],input_sequence_length) 
    y_test = conver_to_lstm_data(y[stop:],output_sequence_length) 
    lstm.train(n_iterations,x_train,y_train)
    pred_lstm= lstm.forward(x_test)
    y_pred = pred_lstm.cpu().detach().numpy().T
    with open('prediction_lstm.txt', 'w') as outfile:
        for slice_2d in y_pred:
            np.savetxt(outfile, slice_2d)
    with open('test_lstm.txt', 'w') as outfile:
        for slice_2d in y_test:
            np.savetxt(outfile, slice_2d)
    
    y_pred.columns = ["x","y","heading","steering_angle","speed","steering_angle_rate"]
    y_pred.to_csv("prediction_lstm.csv")
    
    y_test.to_csv("test_sequences.csv")

    '''
    
    compare_states[:,3]=compare_states[:,3]-compare_states[0,3]
    compare_states[:,0]=compare_states[:,0]-compare_states[0,0]
    compare_states[:,1]=compare_states[:,1]-compare_states[0,1]
    
    states_reg[:,3]=states_reg[:,3]-states_reg[0,3]
    states_reg[:,0]=states_reg[:,0]-states_reg[0,0]
    states_reg[:,1]=states_reg[:,1]-states_reg[0,1]
    y_labels = ["x_position_in_meters","y_position_in_meters","heading"]
    for i in range(compare_states.shape[1]-1):
        plt.figure(figsize=(20,10))
        plt.plot(compare_states[:,3],compare_states[:,i])
        plt.plot(states_reg[:,3],states_reg[:,i])
        plt.legend(["real","prediction"])
        plt.xlabel("time in seconds")
        plt.ylabel(y_labels[i])
        plt.title("Bicycle Model Physics Prediction")
        plt.savefig("physics_model_pred_"+y_labels[i]+".png")
    
    '''
    
    

    

if __name__=="__main__":
    main()
    