import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from bicycle_PINN import PIELM
import datetime
import utm
from process import prep_df_reg
from predict_ode import *
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    sns.set()
    df = pd.read_csv("edited.csv")
    df = df[df["temporaryId"]==df["temporaryId"].loc[0]].reset_index(drop=True)
    test_df=df[df["sub_group"]=='inbound1'].reset_index(drop=True)
    # states_reg = test_ode_reg(test_df)
    # compare_states = np.array(test_df[["x","y","heading","timestamp_posix"]])
    l = test_df['length']
    rho = test_df["steering_angle_rate"]
    x = (test_df["timestamp_posix"]-test_df["timestamp_posix"][0])/(test_df["timestamp_posix"][len(test_df)-1]-test_df["timestamp_posix"][0])
    y = (test_df[["x","y","heading","steering_angle"]]-test_df[["x","y","heading","steering_angle"]].min())/(test_df[["x","y","heading","steering_angle"]].max()-test_df[["x","y","heading","steering_angle"]].min())
    accuracy = 1e-5
    n_iterations = 1e5
    pielm = PIELM(n_nodes=100,input_size= x.shape[0],output_size=y.shape[1])
    pielm.train(accuracy,n_iterations,x,y,l,rho)    
    
    '''
    
    compare_states[:,3]=compare_states[:,3]-compare_states[0,3]
    compare_states[:,0]=compare_states[:,0]-compare_states[0,0]
    compare_states[:,1]=compare_states[:,1]-compare_states[0,1]
    
    states_reg[:,3]=states_reg[:,3]-states_reg[0,3]
    states_reg[:,0]=states_reg[:,0]-states_reg[0,0]
    states_reg[:,1]=states_reg[:,1]-states_reg[0,1]
    y_labels = ["x_position_in_meters","y_position_in_meters","heading"]
    for i in range(compare_states.shape[1]-1):
        plt.figure(figsize=(20,10))
        plt.plot(compare_states[:,3],compare_states[:,i])
        plt.plot(states_reg[:,3],states_reg[:,i])
        plt.legend(["real","prediction"])
        plt.xlabel("time in seconds")
        plt.ylabel(y_labels[i])
        plt.title("Bicycle Model Physics Prediction")
        plt.savefig("physics_model_pred_"+y_labels[i]+".png")
    
    '''
    
    

    

if __name__=="__main__":
    main()
    