import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import train_model,test_model
import sys
from trajectory_prediction.src.bicycle_PINN import Non_Linear_Difference_RNN,PINN_Non_Linear_Difference_RNN,PIELM
import os
import torch
if __name__=="__main__":
    model_type = sys.argv[1]
    train_data = pd.read_csv("data/train.csv")
    input_size = 2 if 'difference' in model_type else 6
    hidden_size = 128
    num_layers = 2
    output_size = 3
    sequence_length = 30
    models_dir = './models/'
    model_path = models_dir+model_type+'.pth'
    if model_type == 'non_linear_difference':
        model = Non_Linear_Difference_RNN(input_size,hidden_size,output_size,num_layers)
    elif 'non_linear_continuous' in model_type:
        model = PIELM(input_size,hidden_size,output_size)
    elif model_type == 'PINN_non_linear_difference':
        model = PINN_Non_Linear_Difference_RNN(input_size,hidden_size,output_size,num_layers)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model "+model_type +" loaded successfully!")
    test_model(model,train_data,model_type,sequence_length)