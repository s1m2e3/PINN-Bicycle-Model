import numpy as np
import pandas as pd
import datetime
import utm



def prep_df_reg(df):
    
    states = ['timestamp_posix','temporaryId','latitude','longitude','speed','heading','length','width','position_on_map','x','y']
    df=df[states]
    df["speed_x"]=df["speed"]*np.cos(df["heading"])
    df["speed_y"]=df["speed"]*np.sin(df["heading"])
    df["accel_x"]=0
    df["accel_y"]=0
    df["ang_speed"]=0
    df["length"]=df["length"]/100
    for id in df["temporaryId"].unique():
        sub_df_index = df[df['temporaryId']==id].index
        df["accel_x"].loc[sub_df_index]=df["speed_x"].loc[sub_df_index].diff()
        df["accel_y"].loc[sub_df_index]=df["speed_y"].loc[sub_df_index].diff()
        df["ang_speed"].loc[sub_df_index]=df["heading"].loc[sub_df_index].diff()

    df["radius"]=df["speed"]/df["ang_speed"]
    df["steering_angle"]=np.arctan((df["length"])/df["radius"])
    df["steering_angle_rate"]=0
    df["steering_angle_rate"]=df["steering_angle"].diff()

    df["sub_group"]=str(0)
    
    
    for id in df["temporaryId"].unique():
        sub_df_index = df[df['temporaryId']==id].index
        df.loc[sub_df_index] = df.loc[sub_df_index].sort_values(by="timestamp_posix")
        
        counter_inbound = 0
        counter_outbound = 0
        counter_inside = 0
        
        
        for index,row in df.loc[sub_df_index].iterrows():
                if index>0:
                    if df["position_on_map"].loc[index]==df["position_on_map"].loc[index-1]:
                        df["sub_group"].loc[index]=df['sub_group'].loc[index-1]
                    elif df["position_on_map"].loc[index]!=df["position_on_map"].loc[index-1]:
                        if df["position_on_map"].loc[index]=="inbound":
                            counter_inbound +=1
                            df["sub_group"].loc[index]= df["position_on_map"].loc[index] +str(counter_inbound)
                        elif df["position_on_map"].loc[index]=="outbound":
                            counter_outbound +=1
                            df["sub_group"].loc[index]=df["position_on_map"].loc[index]+str(counter_outbound)
                        else:
                            counter_inside +=1
                            df["sub_group"].loc[index]=df["position_on_map"].loc[index]+str(counter_inside)

    df=df.fillna(value=0)  
    return df

