import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import datetime

def bicycle_reg(t,u,v,delta,l):
    x,y,theta= u
    dudt =[v*np.cos(theta),v*np.sin(theta),v/l*np.tan(delta)]

    return dudt

def bicycle_lin(t,u,ax,ay,omega):

    x,y,vx,vy,theta= u
    dudt =[vx,vy,ax,ay,omega]

    return dudt


def test_ode_reg(df):
    now = datetime.datetime.now()
    traj_df = df
    traj_df["timestamp_posix"]=traj_df["timestamp_posix"]-traj_df["timestamp_posix"].loc[0]
    u0 = np.array(traj_df[['x','y','heading']].loc[0])
    t=np.array(traj_df["timestamp_posix"].astype(float))
    states=np.zeros((len(t),4))
    states[0]=np.append(u0,t[0])
    for i in range(len(traj_df)):
        if i != len(traj_df)-1:
            v = float(traj_df['speed'].loc[i])
            delta =float(traj_df['steering_angle'].loc[i])
            l = float(traj_df["length"].loc[i])
            sol = solve_ivp(bicycle_reg,t_span=(t[i],t[i+1]),y0=u0,args=(v,delta,l))
            u0=sol.y[:,1]
            states[i+1]=np.append(u0,t[i+1])
    after = datetime.datetime.now()
    #print(states)
    #print("subgroup just finished after:%f",after-now)
    #print("predicted :%f seconds", t[0]-t[-1])

    return states
