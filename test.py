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
    test_data=test_data.drop([idx for idx in range(int(unique_ids[0]),int(unique_ids[0]+200))]).reset_index(drop=True)
    id = test_data['trajectory_id'].reset_index(drop=True)
    # test_data=test_data.drop([idx for idx in range(int(unique_ids[1]),int(unique_ids[1]+20))]).reset_index(drop=True)
    id_ranges,unique_ids = range_trajectories(id)
    print(test_data['trajectory_id'].unique())
    for idx in unique_ids:
        mask = test_data['trajectory_id']==idx
        print(test_data.loc[mask,'timestamp'].shape)
        test_data.loc[mask,'timestamp'] = test_data.loc[mask,'timestamp']-test_data.loc[mask,'timestamp'].loc[id_ranges[idx][0]]
        test_data.loc[mask,'prev_x_coord'] = test_data.loc[mask,'prev_x_coord']-test_data.loc[mask,'prev_x_coord'].loc[id_ranges[idx][0]]
        test_data.loc[mask,'prev_y_coord'] = test_data.loc[mask,'prev_y_coord']-test_data.loc[mask,'prev_y_coord'].loc[id_ranges[idx][0]]
        test_data.loc[mask,'next_x_coord'] = test_data.loc[mask,'next_x_coord']-test_data.loc[mask,'next_x_coord'].loc[id_ranges[idx][0]]
        test_data.loc[mask,'next_y_coord'] = test_data.loc[mask,'next_y_coord']-test_data.loc[mask,'next_y_coord'].loc[id_ranges[idx][0]]

    sub_df = test_data[test_data['trajectory_id']==unique_ids[0]]
    timestamps = sub_df['timestamp'].iloc[1:]
    speed = sub_df['speed'].iloc[0:-1]
    steering_angle = sub_df['steering_angle'].iloc[0:-1]
    x_coord_init = sub_df['prev_x_coord'].iloc[0]
    y_coord_init = sub_df['prev_y_coord'].iloc[0]
    heading_init = sub_df['prev_heading'].iloc[0]
    print(test_data.head())
    x = []
    y = []
    heading = []
    for i,enum in enumerate(speed):
        x.append(x_coord_init+enum*np.cos(heading_init)*0.1)
        y.append(y_coord_init+enum*np.sin(heading_init)*0.1)
        heading.append(heading_init+np.tan(steering_angle[i])*enum/sub_df['length'].iloc[0]*0.1)
        x_coord_init = x[-1]
        y_coord_init = y[-1]
        heading_init = heading[-1]

    next_x_coord = sub_df['next_x_coord'].iloc[0:-1]
    next_y_coord = sub_df['next_y_coord'].iloc[0:-1]
    next_heading = sub_df['next_heading'].iloc[0:-1]
    fig,axes = plt.subplots(3,1,sharex=True)
    fig.suptitle("Vehicle States Predictions", fontsize=20, weight='bold')
    fig.set_figwidth(20)
    fig.set_figheight(10)
    axes[0].plot(timestamps,next_x_coord,label='Sensed data',color='red')
    axes[0].scatter(timestamps,x,label='Physics Model',color='orange',alpha=0.8,s=15)
    
    axes[0].set_ylabel('X coordinate (m)',fontsize=16,style='italic')
    axes[1].plot(timestamps,next_y_coord,label='Sensed data',color='red')
    axes[1].scatter(timestamps,y,label='Physics Model',color='orange',alpha=0.8,s=15)
    axes[1].set_ylabel('Y coordinate (m)',fontsize=16,style='italic')
    axes[2].plot(timestamps,next_heading,label='Sensed data',color='red')
    axes[2].scatter(timestamps,heading,label='Physics Model',color='orange',alpha=0.8,s=15)
    axes[2].set_ylabel('Heading (rad)',fontsize=16,style='italic')
    axes[2].set_xlabel("Time(s)",fontsize=16, style='italic')
    
    axes[2].set_xlim(timestamps.min()-0.5, timestamps.min()+5)
    axes[2].set_ylim(-6.28, 6.28)
    axes[0].set_facecolor('snow')
    axes[1].set_facecolor('snow')
    axes[2].set_facecolor('snow')
    
    for ax in axes:
        ax.grid(color='k',  linewidth=0.5,alpha=0.3)

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
        print(test_data.head())
        predictions[model_type] = test_model(model,test_data,model_type)

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
                    label = 'PINN GRU'
                else:
                    color = 'slateblue'
                    label = 'GRU'
            if 'continuous' not in model_type:
                ax.scatter(timestamps,predictions[model_type][unique_ids[0]][:,i],alpha=0.5,label=label,marker=marker,c=color)
        
    axes[2].legend(loc='lower right')
    fig.savefig('./images/test.png')
    plt.show()
    
