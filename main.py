import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from bicycle_PINN import bicycle_PINN
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
    states_reg = test_ode_reg(test_df)
    compare_states = np.array(test_df[["x","y","heading","timestamp_posix"]])
    print(np.sum(np.power(states_reg[:,0:2]-compare_states[:,0:2],2)))
    pinn_reg = bicycle_PINN(test_df,"reg")
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
    