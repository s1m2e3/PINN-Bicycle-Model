import numpy as np
import pandas as pd

def prep_df(df):
    print(df.columns)
    states = ['timestamp_posix','temporaryId','latitude','longitude','speed','heading','length','width']
    df=df[states]
    df["accel"]=0
    df["accel"]=df["speed"].diff()
    df["ang_speed"]=0
    df["ang_speed"]=df["ang_speed"].diff()
    df["radius"]=df["speed"]/df["ang_speed"]
    df["steering_angle"]=np.arctan(df["length"]/df["radius"])
    df["steering_angle_rate"]=0
    df["steering_angle_rate"]=df["steering_angle"].diff()
    return df

def main():
    df = pd.read_csv("daisy-anthem_remoteBsmLog_06092021_000000.csv")
    df = prep_df(df)
    #pinn = bicycle_PINN(df)


if __name__=="__main__":
    main()
    