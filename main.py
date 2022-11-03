import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import datetime
#import bicycle_PINN
import utm

def bicycle_reg(t,u,a,delta_t,l):
    x,y,v,theta,delta = u
    dudt =[v*np.cos(theta),v*np.sin(theta),a,v/l*np.tan(delta),delta_t]

    return dudt

def bicycle_lin(t,u,ax,ay,omega):

    x,y,vx,vy,theta= u
    dudt =[vx,vy,ax,ay,omega]

    return dudt


def test_ode_reg(df):
    states = {}
    for id in df["temporaryId"].unique():
        states[id]={}
        sub_df = df[df['temporaryId']==id]
        sub_df = sub_df.sort_values(by="timestamp_posix").reset_index(drop=True)
        
        counter_inbound = 0
        counter_outbound = 0
        counter_inside = 0
        
        sub_df['sub_group']=str(0)
        for index,row in sub_df.iterrows():
                if index>0:
                    if sub_df["position_on_map"].loc[index]==sub_df["position_on_map"].loc[index-1]:
                        sub_df["sub_group"].loc[index]=sub_df['sub_group'].loc[index-1]
                    elif sub_df["position_on_map"].loc[index]!=sub_df["position_on_map"].loc[index-1]:
                        if sub_df["position_on_map"].loc[index]=="inbound":
                            counter_inbound +=1
                            sub_df["sub_group"].loc[index]= sub_df["position_on_map"].loc[index] +str(counter_inbound)
                        elif sub_df["position_on_map"].loc[index]=="outbound":
                            counter_outbound +=1
                            sub_df["sub_group"].loc[index]=sub_df["position_on_map"].loc[index]+str(counter_outbound)
                        else:
                            counter_inside +=1
                            sub_df["sub_group"].loc[index]=sub_df["position_on_map"].loc[index]+str(counter_inside)
        if "accel_x" not in df.columns:
            for subgroup in sub_df["sub_group"].unique():
                now = datetime.datetime.now()
                traj_df = sub_df[sub_df['sub_group']==subgroup].reset_index(drop=True)
                traj_df["timestamp_posix"]=traj_df["timestamp_posix"]-traj_df["timestamp_posix"].loc[0]
                u0 = np.array(traj_df[['x','y','speed','heading','steering_angle']].loc[0])
                t=np.array(traj_df["timestamp_posix"].astype(float))
                states[id][subgroup]=np.zeros((len(t),6))
                states[id][subgroup][0]=np.append(u0,t[0])
                for i in range(len(traj_df)):
                    if i != len(traj_df)-1:
                        a = float(traj_df['accel'].loc[i])
                        delta_t =float(traj_df['steering_angle_rate'].loc[i])
                        l = float(traj_df["length"].loc[i])
                        sol = solve_ivp(bicycle_reg,t_span=(t[i],t[i+1]),y0=u0,args=(a,delta_t,l))
                        u0=sol.y[:,1]
                        states[id][subgroup][i+1]=np.append(u0,t[i+1])
                after = datetime.datetime.now()
                print("subgroup just finished after:%f",after-now)
                print("predicted :%f seconds", t[0]-t[-1])
        else:
            for subgroup in sub_df["sub_group"].unique():
                now = datetime.datetime.now()
                traj_df = sub_df[sub_df['sub_group']==subgroup].reset_index(drop=True)
                traj_df["timestamp_posix"]=traj_df["timestamp_posix"]-traj_df["timestamp_posix"].loc[0]
                u0 = np.array(traj_df[['x','y','speed_x',"speed_y",'heading']].loc[0])
                t = np.array(traj_df["timestamp_posix"].astype(float))
                states[id][subgroup]=np.zeros((len(t),6))
                states[id][subgroup][0]=np.append(u0,t[0])
                for i in range(len(traj_df)):
                    if i != len(traj_df)-1:
                        a_x = float(traj_df['accel_x'].loc[i])
                        a_y = float(traj_df['accel_y'].loc[i])
                        omega =float(traj_df['ang_speed'].loc[i])
                        sol = solve_ivp(bicycle_lin,t_span=(t[i],t[i+1]),y0=u0,args=(a_x,a_y,omega))
                        u0=sol.y[:,1]
                        states[id][subgroup][i+1]=np.append(u0,t[i+1])
                after = datetime.datetime.now()
                print("subgroup just finished after:%f",after-now)
                print("predicted :%f seconds", t[0]-t[-1])

    return states

#def test_ode_lin(df):

    
#    return MSE


def prep_df_reg(df):
    states = ['timestamp_posix','temporaryId','latitude','longitude','speed','heading','length','width','position_on_map','x','y']
    df=df[states]
    df["accel"]=0
    df["accel"]=df["speed"].diff()
    df["ang_speed"]=0
    df["ang_speed"]=df["heading"].diff()
    df["radius"]=df["speed"]/df["ang_speed"]
    df["steering_angle"]=np.arctan(df["length"]/df["radius"])
    df["steering_angle_rate"]=0
    df["steering_angle_rate"]=df["steering_angle"].diff()
    df=df.fillna(value=0)  
    return df

def prep_df_lin(df):
    states = ['timestamp_posix','temporaryId','latitude','longitude','speed','heading','length','width','position_on_map','x','y']
    df=df[states]
    df["speed_x"]=df["speed"]*np.cos(df["heading"])
    df["speed_y"]=df["speed"]*np.sin(df["heading"])
    df["accel_x"]=0
    df["accel_y"]=0
    df["ang_speed"]=0
    
    for id in df["temporaryId"].unique():
        sub_df_index = df[df['temporaryId']==id].index
        df["accel_x"].loc[sub_df_index]=df["speed_x"].loc[sub_df_index].diff()
        df["accel_y"].loc[sub_df_index]=df["speed_y"].loc[sub_df_index].diff()
        df["ang_speed"].loc[sub_df_index]=df["heading"].loc[sub_df_index].diff()

    df=df.fillna(value=0)    
    
    return df


def main():
    df = pd.read_csv("daisy-anthem_remoteBsmLog_06092021_000000.csv")
    (x,y,_,_)=utm.from_latlon(np.array(df['latitude']),np.array(df['longitude']))
    df['x']=x
    df['y']=y
    df_lin = prep_df_lin(df)
    df_reg = prep_df_reg(df)
    states_reg = test_ode_reg(df_reg)
    states_lin =test_ode_reg(df_lin)
    #df_reg = prep_df_reg(df)
    #pinn = bicycle_PINN(df_lin)


if __name__=="__main__":
    main()
    