import numpy as np
import pandas as pd
import random
import os
import torch
from utils import train_model,test_model
import sys
from trajectory_prediction.src.bicycle_PINN import Difference_RNN,PINN_Difference_RNN,Non_Linear_Difference_RNN,PINN_Non_Linear_Difference_RNN
from trajectory_prediction.src.model import LSTM


if __name__=="__main__":
    model_type = sys.argv[1]
    data = pd.read_csv("data/edited_trajectory.csv")
    sequence_length=50
    trajectories = []
    for id in data['id'].unique():
        sub_df = data[data['id']==id]
        for trajectory in sub_df['trajectory'].unique():
            if len(sub_df[sub_df['trajectory']==trajectory])>=sequence_length:
                trajectories.append(sub_df[sub_df['trajectory']==trajectory])
    random.shuffle(trajectories)
    train_len = int(len(trajectories)*0.7)
    train = trajectories[:train_len]
    test = trajectories[train_len:]
    output_size = 2
    hidden_size = 32
    matrix_A_shape = (2,2)
    matrix_B_shape = (2,2)
    input_size = 4
    num_layers = 2
    models_dir = './models/'
    model_path = models_dir+model_type+'.pth'

    if model_type == 'non_linear_continuous':
        pass
    elif model_type == 'non_linear_difference' or model_type == 'PINN_non_linear_difference':
        if model_type == 'PINN_non_linear_difference':
            model = PINN_Non_Linear_Difference_RNN(input_size,hidden_size,output_size)
        else:
            model = Non_Linear_Difference_RNN(input_size,hidden_size,output_size)
        
        if os.path.exists(model_path):
            # Instantiate the model
            # Load the model from the file
            model.load_state_dict(torch.load(model_path))
            print("Model "+model_type +" loaded successfully!")
        else:
            print(f"The file {model_path} does not exist.")

    elif model_type == 'linear_difference' or model_type == 'PINN_linear_difference':
        
        if model_type == 'PINN_linear_difference':
            model = PINN_Difference_RNN(matrix_A_shape,matrix_B_shape)
        else:
            model = Difference_RNN(matrix_A_shape,matrix_B_shape)
        if os.path.exists(model_path):
            # Instantiate the model
            # Load the model from the file
            model.load_state_dict(torch.load(model_path))
            print("Model "+model_type +" loaded successfully!")
        else:
            print(f"The file {model_path} does not exist.")
        
    elif model_type == 'lstm':
        pass
        # model = LSTM(input_size, hidden_size, num_layers, output_size,input_sequence_length,output_sequence_length)
    else:
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    model = train_model(model,train,model_type,sequence_length)
    # predict = test_model(model,test,model_type,sequence_length)
    # predict = np.array(predict)
    # with open("./data/predictions_" + model_type + "_" + str(sequence_length) + ".csv", "w") as f:
    #     np.savetxt(f, predict, delimiter=",", fmt='%s')
