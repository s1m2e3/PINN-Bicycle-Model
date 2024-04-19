import numpy as np
import pandas as pd
import copy
import torch.nn as nn
import torch.optim as optim
import torch 
import matplotlib.pyplot as plt
import random
import time

def train_loop(x,u,y,timedeltas,model,model_type,sub_sequence):
    x = torch.from_numpy(x).float().to(model.device)
    u = torch.from_numpy(u).float().to(model.device)
    y = torch.from_numpy(y).float().to(model.device)
    timedeltas = torch.from_numpy(timedeltas).float().to(model.device)
    if 'non_linear' not in model_type:
        x = torch.reshape(x,(1,len(x)))
        u = torch.transpose(u,0,1)
    elif 'non_linear' in model_type:
        x_pass = x[:,:,0].reshape(x.shape[0],x.shape[1],1)
        timedeltas = timedeltas.reshape(timedeltas.shape[0],timedeltas.shape[1],1)
        u = torch.concat((u,x_pass),axis=2)
        u = torch.concat((u,timedeltas),axis=2)
    loss = nn.MSELoss()
    clip_value = 0.1
    lr = 1e-3
    model.to(model.device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    sub_set = random.randint(0, x.shape[0]-sub_sequence)
    time = time.time()
    for i in range(10000):
        optimizer.zero_grad()
        x_out = model.forward(x,u)
        output = loss(x_out[sub_set:sub_set+sub_sequence,:,:],y[sub_set:sub_set+sub_sequence,:,:])
        if  i==9999:
            plt.figure(figsize=(20,10))
            plt.subplot(131)
            plt.plot(x_out.detach().cpu().numpy()[:,0,0])
            plt.plot(y.detach().cpu().numpy()[:,0,0])
            plt.subplot(132)
            plt.plot(x_out.detach().cpu().numpy()[:,0,1])
            plt.plot(y.detach().cpu().numpy()[:,0,1])
            plt.subplot(133)
            plt.plot(x_out.detach().cpu().numpy()[:,0,0],x_out.detach().cpu().numpy()[:,0,1])
            plt.plot(y.detach().cpu().numpy()[:,0,0],y.detach().cpu().numpy()[:,0,1])
            plt.show()
        if i % 1000 == 0:
            print(f"Epoch {i+1}: Loss = {output:.6f}")
        if 'PINN' in model_type:
            x_out = model.forward_PINN(x,u,timedeltas)
            output+= loss(x_out,torch.zeros(x_out.shape).to(model.device))
        if output.item() < 1e-3:
            break
        output.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
    
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

def train_model(model,train_data,model_type,sequence_length,sub_sequence):
    
    for trajectory in train_data[0:5]:
        if model_type == 'PINN_linear_difference' or model_type == 'linear_difference':   
            x = trajectory['timestamp']
            y = trajectory[['x_coord','y_coord']]
            u = trajectory[['speed','heading']]
            timedeltas = trajectory['delta_timestamps']
            prep = prepare_data_for_difference(x,u,y,timedeltas,sequence_length,model_type)
            for row in prep:
                x = prep[row][0]
                u = prep[row][1]
                y = prep[row][2]
                timedeltas = prep[row][3]
                train_loop(x,u,y,timedeltas,model,model_type,sub_sequence)

        elif model_type == 'non_linear_difference' or model_type == 'PINN_non_linear_difference':
            x = trajectory['timestamp']
            y = trajectory[['x_coord','y_coord']]
            u = trajectory[['speed','heading']]
            timedeltas = trajectory['delta_timestamps']
            prep = prepare_data_for_recurrent(x,u,y,timedeltas,sequence_length)
            x = prep[0]
            u = prep[1]
            y = prep[2]
            timedeltas = prep[3]
            train_loop(x,u,y,timedeltas,model,model_type,sub_sequence)

        elif model_type == 'lstm':
            x,u,y = prepare_data_for_recurrent(train_data[trajectory][0],train_data[trajectory][1],train_data[trajectory][2],sequence_length,model_type)
            model.train(x,y,sequence_length)
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
