import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import test_model,remove_trajectory,range_trajectories
import sys
from trajectory_prediction.src.bicycle_PINN import GRU,PIELM
import os
import torch
import seaborn as sns

if __name__=="__main__":
    sns.set_theme()
    model_types = ['non_linear_continuous','non_linear_difference','PINN_non_linear_difference','PINN_non_linear_continuous']
    test_data = pd.read_csv("data/test.csv")
    id = test_data['trajectory_id'].reset_index(drop=True)
    id_ranges,unique_ids = range_trajectories(id)
    predictions = {}
    test_data = remove_trajectory(test_data)
    for model_type in model_types:
        input_size = 6 if 'non_linear_continuous' in model_type else 2
        hidden_size = 256 if 'non_linear_continuous' in model_type else 32
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
        predictions[model_type] = test_model(model,test_data,model_type)

sub_df = test_data[test_data['trajectory_id']==unique_ids[1]]
timestamps = sub_df['timestamp'].iloc[1:]
next_x_coord = sub_df['next_x_coord'].iloc[0:-1]
next_y_coord = sub_df['next_y_coord'].iloc[0:-1]
next_heading = sub_df['next_heading'].iloc[0:-1]
fig,axes = plt.subplots(3,1,sharex=True)
fig.suptitle("Predictions by models")
fig.set_figwidth(20)
fig.set_figheight(10)
axes[0].plot(timestamps,next_x_coord,label='Sensed data',color='green')
axes[0].set_xlabel("Time(s)")
axes[0].set_ylabel("X coordinate (m)")
axes[1].plot(timestamps,next_y_coord,label='Sensed data',color='green')
axes[1].set_ylabel("Y coordinate (m)")
axes[2].plot(timestamps,next_heading,label='Sensed data',color='green')
axes[2].set_ylabel("Heading angle (rad)")

    
for i in range(3):
    ax = axes[i]
    for j,model_type in enumerate(model_types):
        if 'continuous' in model_type:
            marker = 'v'
            if 'PINN' in model_type:
                color = 'darkorange'
            else:
                color = 'goldenrod'
        else:
            
            marker = 'x'
            if 'PINN' in model_type:
                color = 'navy'
            else:
                color = 'slateblue'
        if 'continuous' not in model_type:  
            ax.scatter(timestamps,predictions[model_type][unique_ids[1]][:,i],alpha=0.5,label=model_type,marker=marker,c=color)
    ax.set_xlim(-0.05,3)
    ax.legend(loc='lower right')
    
plt.show()
   
