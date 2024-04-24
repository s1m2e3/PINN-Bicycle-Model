import numpy as np
import pandas as pd
import copy
import torch.nn as nn
import torch.optim as optim
import torch 
import matplotlib.pyplot as plt
import random
import time
import csv
def train_loop(x,y,model,model_type,id):
    unique_ids = id.unique()
    id_ranges = {}
    for uid in unique_ids:
        start_idx = id[id==uid].index[0]
        end_idx = id[id==uid].index[-1]
        id_ranges[uid] = (start_idx, end_idx)
    x = torch.from_numpy(x).double().to(model.device)
    y = torch.from_numpy(y).double().to(model.device)
    loss = nn.MSELoss()
    
    if 'difference' in model_type or 'recurrent' in model_type:
        clip_value = 0.01
    lr = 1e-2
    model.to(model.device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for i in range(10000):
        output = 0
        optimizer.zero_grad()
        if 'recurrent' not in model_type and 'difference' not in model_type:
            x_out = model.forward(x)
            output = loss(x_out,y)
            if i  %10 == 0:
                print(f"Epoch {i+1}: Loss = {output:.6f}")
        if 'difference' in model_type:
            difference_loss = 0
            for uid in unique_ids:
                x_out = model.forward(x[id_ranges[uid][0]:id_ranges[uid][1]])
                y_out = y[id_ranges[uid][0]:id_ranges[uid][1]].reshape(x_out.shape)
                difference_loss += loss(x_out,y_out)
            output+= difference_loss 
            if i  %10 == 0:
                print(f"Epoch {i+1}: Loss = {output:.6f}")
        if 'PINN' in model_type:
            x_out = model.forward_PINN(x)
            phis_loss = loss(x_out,torch.zeros(x_out.shape,dtype=torch.double).to(model.device))
            if i  %10 == 0:
                print(f"Epoch {i+1} Difference Equation: Loss = {phis_loss:.6f}")
            output+= phis_loss
        
        if 'recurrent' in model_type:
            recurrent_loss = 0
            for uid in unique_ids:
                x_out = model.forward_recurrent(x[id_ranges[uid][0]:id_ranges[uid][1]])
                recurrent_loss += loss(x_out,y[id_ranges[uid][0]:id_ranges[uid][1]])
                
            if i  %10 == 0:
                print(f"Epoch {i+1} in Recurrence: Loss = {recurrent_loss:.6f}")
            output+= recurrent_loss    
        if output.item() < 1e-3:
            break
        output.backward()
        # if 'difference' in model_type or 'recurrent' in model_type:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        if i %100 == 0:
            torch.save(model.state_dict(), './models/'+model_type+'.pth')
            
    

def numerical_derivative(f,timestamps,timestamps_delta):
    f=np.array(f)
    timestamps=np.array(timestamps)
    timestamps_delta=np.array(timestamps_delta)
    x = np.arange(len(f))
    num_der = np.zeros(len(f))
    num_der_indices = np.arange(1, len(f)-1)
    for i in num_der_indices:
        num_der[i] = (f[i + 1] - f[i - 1]) / (timestamps_delta[i]+timestamps_delta[i+1])
    return num_der
def prepare_data_for_difference(x,u,y,timedeltas,sequence_length,model_type):
    prep_data = {}
    x = pd.concat((x,y),axis=1)
    x = np.array(x)
    u = np.array(u)
    y = np.array(y)
    timedeltas = np.array(timedeltas)
    x = x[0:-1,:]
    y = y[1:,:]
    u = u[0:-1,:]
    
    timedeltas = timedeltas[1:]
    # if model_type == 'non_linear_difference':
    #     for i in range(len(u)):
    #         prep_data[i] = [x[i,:],u[i:i+sequence_length,:],y[i:i+sequence_length,:]]
    # elif model_type == 'linear_difference':
    x=x[:,1:]
    for i in [0]:
        prep_data[i] = [x[i,:],u[i:i+sequence_length,:],y[i:i+sequence_length,:],timedeltas[i:i+sequence_length]]
    return prep_data

def prepare_data_for_continuous(x,u,y):
    prep_data = {}
    x = np.array(x)
    u = np.array(u)
    y = np.array(y)
    x = np.concatenate((x,u),axis=1)
    return x,y

def prepare_data_for_recurrent(x,u,y,timedeltas,sequence_length):
    x = pd.concat((x,y),axis=1)
    x = np.array(x)
    u = np.array(u)
    y = np.array(y)
    timedeltas = np.array(timedeltas)
    x = x[0:-1,:]
    y = y[1:,:]
    u = u[0:-1,:]
    timedeltas = timedeltas[1:]
    new_x = np.zeros((sequence_length,x.shape[0]-sequence_length,x.shape[1]))
    new_u = np.zeros((sequence_length,u.shape[0]-sequence_length,u.shape[1]))
    new_y = np.zeros((sequence_length,y.shape[0]-sequence_length,y.shape[1]))
    new_timedeltas = np.zeros((sequence_length,timedeltas.shape[0]-sequence_length))
    for i in range(x.shape[0]-sequence_length-1):
        new_x[:,i,:]=x[i:i+sequence_length,:]
        new_u[:,i,:]=u[i:i+sequence_length,:]
        new_y[:,i,:]=y[i:i+sequence_length,:]
        new_timedeltas[:,i]=timedeltas[i:i+sequence_length]
    
    return [new_x,new_u,new_y,new_timedeltas]

def remove_trajectory(train_data):
    df = pd.DataFrame({'timestamp':[],'prev_x_coord':[],'prev_y_coord':[],'prev_heading':[],'speed':[],'steering_angle':[],'delta_timestamps':[],
                       'next_x_coord':[],'next_y_coord':[],'next_heading':[],'trajectory_id':[],'length':[]})
    for trajectory in train_data['trajectory_id'].unique():
        sub_trajectory = train_data[train_data['trajectory_id']==trajectory]
        length = sub_trajectory['length'].unique()[0]
        x = sub_trajectory[['timestamp','delta_timestamps','x_coord','y_coord','heading','speed','steering_angle']]
        x = np.array(x)
        x = x[0:-1,:]
        y = np.array(sub_trajectory[['x_coord','y_coord','heading']])
        y = y[1:,:]
        x = pd.DataFrame(np.concatenate((x,y),axis=1))
        x.columns = ['timestamp','delta_timestamps','prev_x_coord','prev_y_coord','prev_heading','speed','steering_angle','next_x_coord','next_y_coord','next_heading']
        x['trajectory_id'] = trajectory
        x['length'] = length
        df = pd.concat((df,x),axis=0)
    return df

def train_model(model,train_data,model_type,sequence_length):
    train_data = remove_trajectory(train_data)
    length = train_data['length']
    x = train_data[['timestamp','speed','steering_angle','prev_x_coord','prev_y_coord','prev_heading']]
    y = train_data[['next_x_coord','next_y_coord','next_heading']]
    id = train_data['trajectory_id'].reset_index(drop=True)
    model.set_length(length)
    if 'recurrent' in model_type:
            model.set_recurrent(True)
    
    x = np.array(x)
    y = np.array(y)
    train_loop(x,y,model,model_type,id)

    return model

def test_model(model,test,model_type,sequence_length):
    predictions = []
    for trajectory in test:
        x = trajectory['timestamp']
        y = trajectory[['x_coord','y_coord','heading']]
        u = trajectory[['speed','angle']]
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
