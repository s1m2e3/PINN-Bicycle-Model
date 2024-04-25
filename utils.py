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
def range_trajectories(id):
    unique_ids = id.unique()
    id_ranges = {}
    for uid in unique_ids:
        start_idx = id[id==uid].index[0]
        end_idx = id[id==uid].index[-1]
        id_ranges[uid] = (start_idx, end_idx)
    return id_ranges, unique_ids
def train_loop(x,y,model,model_type,id):
    id_ranges, unique_ids = range_trajectories(id)
    x = torch.from_numpy(x).double().to(model.device)
    y = torch.from_numpy(y).double().to(model.device)
    loss = nn.MSELoss()
    
    lr = 1e-3
    model.to(model.device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for i in range(10000):
        output = 0
        optimizer.zero_grad()
        if 'difference' not in model_type:
            x_out = model.forward(x)
            output = loss(x_out,y)
            if i  %100 == 0:
                print(f"Epoch {i+1}: Loss = {output:.6f}")
        if 'difference' in model_type:
            difference_loss = 0
            for uid in unique_ids:
                x_out = model.forward(x[id_ranges[uid][0]:id_ranges[uid][1]])
                y_out = y[id_ranges[uid][0]:id_ranges[uid][1]].reshape(x_out.shape)
                difference_loss += loss(x_out,y_out)
            output+= difference_loss 
            if i  %100 == 0:
                print(f"Epoch {i+1}: Loss = {output:.6f}")
        if 'PINN' in model_type:
            x_out = model.forward_PINN(x)
            phis_loss = loss(x_out,torch.zeros(x_out.shape,dtype=torch.double).to(model.device))
            if i  %100 == 0:
                print(f"Epoch {i+1} Difference Equation: Loss = {phis_loss:.6f}")
            output+= phis_loss
        
        if output.item() < 1e-3:
            break
        output.backward()
        optimizer.step()
        if i %100 == 0:
            torch.save(model.state_dict(), './models/'+model_type+'.pth')


def normalize_x(x):
    x_max = np.max(x,axis=0)
    x_min = np.min(x,axis=0)
    normalizing_factor = (2) / (x_max - x_min)
    x_norm = -1+normalizing_factor*(x-x_min)
    x = x_norm
    return x_norm,normalizing_factor,x_min

def train_model(model,train_data,model_type):
    
    length = train_data['length']
    model.set_length(length)
    x = train_data[['timestamp','speed','steering_angle','prev_x_coord','prev_y_coord','prev_heading']]
    y = train_data[['next_x_coord','next_y_coord','next_heading']]
    id = train_data['trajectory_id'].reset_index(drop=True)
    x = np.array(x)
    y = np.array(y)
    x,normalizing_factor,x_min = normalize_x(x)
    model.set_normalizing_factor(normalizing_factor)
    model.set_x_min(x_min)
    train_loop(x,y,model,model_type,id)

def test_model(model,test_data,model_type):
    
    x = test_data[['timestamp','speed','steering_angle','prev_x_coord','prev_y_coord','prev_heading']]
    id = test_data['trajectory_id'].reset_index(drop=True)
    x = np.array(x)
    x,normalizing_factor,x_min = normalize_x(x)
    id_ranges,unique_ids = range_trajectories(id)
    x = torch.from_numpy(x).double().to(model.device)
    predictions = {}
    for uid in unique_ids:
        if 'continuous' not in model_type:
            x_out = model.forward(x[id_ranges[uid][0]:id_ranges[uid][1]])
            x_out = x_out.reshape(x[id_ranges[uid][0]:id_ranges[uid][1]].shape[0],3).cpu().detach().numpy()
            predictions[uid] = x_out
            
        else:
            x_out_stacked = model.forward_recurrent(x[id_ranges[uid][0]:id_ranges[uid][0]+1])
            x_out = x_out_stacked
            for i in range(id_ranges[uid][0]+1,id_ranges[uid][1]):
                controls = x[i,0:3].reshape(1,3)
                x_forward = torch.cat((controls,x_out),1)
                x_out = model.forward(x_forward)
                x_out_stacked = torch.cat((x_out_stacked,x_out),0)
            predictions[uid] = x_out_stacked.cpu().detach().numpy()
            
    return predictions