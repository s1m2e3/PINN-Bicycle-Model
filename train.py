import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import train_model,test_model
import sys
from trajectory_prediction.src.bicycle_PINN import Non_Linear_Difference_RNN,PINN_Non_Linear_Difference_RNN,PIELM


if __name__=="__main__":
    model_type = sys.argv[1]
    train_data = pd.read_csv("data/train.csv")
    input_size = 2 if 'difference' in model_type else 6
    hidden_size = 128
    num_layers = 2
    output_size = 3
    sequence_length = 30
    
    if model_type == 'non_linear_difference':
        model = Non_Linear_Difference_RNN(input_size,hidden_size,output_size,num_layers)
    elif model_type == 'non_linear_continuous':
        model = PIELM(input_size,hidden_size,output_size,None)
    elif model_type == 'PINN_non_linear_difference':
        model = PINN_Non_Linear_Difference_RNN(input_size,hidden_size,output_size,num_layers)
    train_model(model,train_data,model_type,sequence_length)

