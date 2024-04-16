import numpy as np
import pandas as pd
import random
import copy 
import sys
from trajectory_prediction.src.bicycle_PINN import Difference_RNN,Non_Linear_Difference_RNN
from trajectory_prediction.src.model import LSTM
def prepare_data_for_difference(x,u,y,sequence_length,model_type):
    prep_data = {}
    x = np.concat((x,y),axis=1)
    x = np.array(x)
    u = np.array(u)
    y = np.array(y)
    x = x[0:-1,:]
    y = y[1:,:]
    u = u[0:-1,:]
    if model_type == 'non_linear_difference':
        for i in range(len(u)):
            prep_data[i] = [x[i,:],u[i:i+sequence_length,:],y[i:i+sequence_length,:]]
    elif model_type == 'linear_difference':
        x=x[:,1:]
        for i in range(len(u)):
            prep_data[i] = [x[i,:],u[i:i+sequence_length,:],y[i:i+sequence_length,:]]
    return prep_data

def prepare_data_for_continuous(x,u,y):
    prep_data = {}
    x = np.array(x)
    u = np.array(u)
    y = np.array(y)
    x = np.concatenate((x,u),axis=1)
    return x,y

def prepare_data_for_recurrent(x,u,y,sequence_length):

    x = np.array(x)
    u = np.array(u)
    y = np.array(y)
    y = copy.deepcopy(x)
    x = x[0:-1,:]
    y = y[1:,:]
    u = u[0:-1,:]
    new_x = np.zeros((x.shape[0],sequence_length,x.shape[1]))
    new_u = np.zeros((u.shape[0],sequence_length,u.shape[1]))
    new_y = np.zeros((y.shape[0],sequence_length,y.shape[1]))
    for i in range(len(u)):
        for j in range(len(sequence_length)):    
            new_x[i,j,:]=x[i,:]
            new_u[i,j,:]=u[i,:]
            new_y[i,j,:]=y[i,:]
    return new_x,new_u,new_y

def train_model(model,train_data,model_type,sequence_length):
    for trajectory in train_data:
        x = trajectory['timestamp']
        y = trajectory['lat','long','heading']
        u = trajectory['speed_x','speed_y','angle']
        if model_type == 'non_linear_continuous':
            x,y = prepare_data_for_continuous(x,u,y)
            model.train(x,y,sequence_length)
        elif model_type == 'non_linear_difference' or model_type == 'linear_difference':
            prep = prepare_data_for_difference(x,u,y,sequence_length,model_type)
            for row in prep:
                x = prep[row][0]
                u = prep[row][1]
                y = prep[row][2]
                model.train(1e-3,1e2,x,u,y)
        elif model_type == 'lstm':
            x,u,y = prepare_data_for_recurrent(train_data[trajectory][0],train_data[trajectory][1],train_data[trajectory][2],sequence_length,model_type)
            model.train(x,y,sequence_length)
    return model

def test_model(model,test,model_type,sequence_length):
    predictions = []
    for trajectory in test:
        x = trajectory['timestamp']
        y = trajectory['lat','long','heading']
        u = trajectory['speed_x','speed_y','angle']
        if model_type == 'non_linear_continuous':
            x,y = prepare_data_for_continuous(x,u,y)
            predictions.append(model.forward(x))
        elif model_type == 'non_linear_difference' or model_type == 'linear_difference':
            prep = prepare_data_for_difference(x,u,y,sequence_length,model_type)
            traj_predictions = []
            for row in prep:
                x = prep[row][0]
                u = prep[row][1]
                traj_predictions.append(model.predict(x,u))
            predictions.append(traj_predictions)
        elif model_type == 'lstm':
            x,u,y = prepare_data_for_recurrent(test[trajectory][0],test[trajectory][1],test[trajectory][2],sequence_length,model_type)
            predictions.append(model.forward(x))
    return predictions

if __name__=="__main__":
    model_type = sys.argv[1]
    # sequence_length = int(sys.argv[2])
    data = pd.read_csv("data/edited_trajectory.csv")
    trajectories = []
    for id in data['id'].unique():
        sub_df = data[data['id']==id]
        for trajectory in sub_df['trajectory'].unique():
            trajectories.append(sub_df[sub_df['trajectory']==trajectory])
    random.shuffle(trajectories)
    train_len = int(len(trajectories)*0.7)
    train = trajectories[:train_len]
    test = trajectories[train_len:]
    sequence_length=10
    hidden_size = 32
    matrix_A_shape = (3,3)
    matrix_B_shape = (3,2)
    num_layers = 2
    if model_type == 'non_linear_continuous':
        pass
    elif model_type == 'non_linear_difference':
        model = Non_Linear_Difference_RNN(matrix_A_shape,matrix_B_shape,sequence_length,hidden_size)
    elif model_type == 'linear_difference':
        model = Difference_RNN(matrix_A_shape,matrix_B_shape,sequence_length)
    elif model_type == 'lstm':
        pass
        # model = LSTM(input_size, hidden_size, num_layers, output_size,input_sequence_length,output_sequence_length)
    else:
        sys.exit(1)
    model = train_model(model,train,model_type,sequence_length)
    predict = test_model(model,test,model_type,sequence_length)
    predict = np.array(predict)
    with open("./data/predictions_" + model_type + "_" + str(sequence_length) + ".csv", "w") as f:
        np.savetxt(f, predict, delimiter=",", fmt='%s')
