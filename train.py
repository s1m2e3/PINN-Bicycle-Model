import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import train_model,remove_trajectory
import sys
from trajectory_prediction.src.bicycle_PINN import GRU,PIELM
import os
import torch
if __name__=="__main__":
    model_type = sys.argv[1]
    train_data = pd.read_csv("data/train.csv")
    input_size = 6 if 'non_linear_continuous' in model_type else 2
    hidden_size = 256
    num_layers = 2
    output_size = 3
    sequence_length = 30
    models_dir = './models/'
    model_path = models_dir+model_type+'.pth'
    
    if 'non_linear_continuous' in model_type:
        model = PIELM(input_size,hidden_size,output_size)
    elif 'difference' in model_type:
        model = GRU(input_size,hidden_size,output_size,num_layers)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model "+model_type +" loaded successfully!")
    train_data = remove_trajectory(train_data)
    train_model(model,train_data,model_type,sequence_length)

