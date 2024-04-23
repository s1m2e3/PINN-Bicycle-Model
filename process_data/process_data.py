import sys
sys.path.insert(0, '..')
import pandas as pd
import geopandas as gpd
import numpy as np 
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from utils import numerical_derivative
from scipy.interpolate import splrep, BSpline, splev


def interp_sub_traj(sub_traj_df):
    new_timestamps = np.arange(sub_traj_df['timestamp'].min(),sub_traj_df['timestamp'].max()+0.01,0.1)
    new_sub_traj_df = pd.DataFrame({'timestamp':new_timestamps})
    new_sub_traj_df['length'] = sub_traj_df['length'].iloc[0]
    new_sub_traj_df['delta_timestamps'] = 0.1
    # fig, axes = plt.subplots(4, 1, sharex=True)
    # for i, col in enumerate(['x_coord', 'y_coord', 'heading', 'speed']):
    #     axes[i].scatter(sub_traj_df['timestamp'], sub_traj_df[col],  alpha=1)
    # axes[2].set_ylim(-6.28, 6.28) # for heading
    # axes[3].set_ylim(0, 40) # for speed
    for i in ['x_coord', 'y_coord', 'heading', 'speed']:
        if i == 'timestamp':
            pass
        else:
            try:
                tck = splrep(sub_traj_df['timestamp'],sub_traj_df[i],k=1,s=5)
            except RuntimeWarning as e:
                print(f"Runtime warning for column {i}: {e}")
                raise RuntimeWarning
            

            new_y = splev(new_timestamps,tck, der = 0)
            if np.isnan(new_y).any():
                raise RuntimeWarning
            new_sub_traj_df[i] = new_y
            
    sub_traj_df = new_sub_traj_df
    trajectory = sub_traj_df
    # for i, col in enumerate(['x_coord', 'y_coord', 'heading', 'speed']):
    #     axes[i].scatter(trajectory['timestamp'], trajectory[col],  alpha=0.5)
    #     print(trajectory[col])
    # plt.show()
    
    return sub_traj_df

df = pd.read_csv('../data/edited.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
df['timestamp_date']=pd.to_datetime(df['timestamp'],unit='s').dt.minute
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat))
gdf.crs = "EPSG:4326"
gdf = gdf.to_crs(epsg=3086)
gdf['x_coord'] = gdf['geometry'].x
gdf['y_coord'] = gdf['geometry'].y
unique_ids = np.unique(gdf['id'])
gdf['trajectory']=0
gdf['delta_timestamps']=0
gdf['angular_speed']=0  
gdf['angle']=0
gdf['relevant_trajectory']=0
gdf['steering_angle']=0
mask_1 = gdf['heading']<90
mask_2 = gdf['heading']>90

gdf.loc[mask_1,'heading'] = 90-gdf.loc[mask_1,'heading']
gdf.loc[mask_2,'heading'] = -(gdf.loc[mask_2,'heading']-90)

gdf['heading'] = gdf['heading']*np.pi/180.0

for unique_id in unique_ids:
    sub_df = gdf[gdf['id']==unique_id]
    for i in range(len(sub_df)-1):
        if sub_df['timestamp'].iloc[i+1]-sub_df['timestamp'].iloc[i]>2:
            sub_df['trajectory'].iloc[i+1]=sub_df['trajectory'].iloc[i]+1
        else:
            sub_df['trajectory'].iloc[i+1]=sub_df['trajectory'].iloc[i]
    gdf[gdf['id']==unique_id]=sub_df

traj = pd.DataFrame({'timestamp':[],'x_coord':[],'y_coord':[],'heading':[],'speed':[],'length':[],'angular_speed':[],'steering_angle':[]})
traj_count = 0
for id in unique_ids:
    sub_plot = gdf[gdf['id']==id]
    lines = []
    
    for trajectory in sub_plot['trajectory'].unique():
        
        sub_traj_df = sub_plot[sub_plot['trajectory']==trajectory]
        if len(sub_traj_df)>30:
            try:
                sub_traj_df = interp_sub_traj(sub_traj_df)
                
            except RuntimeWarning:
                print('cant interpolate')
            sub_traj_df['x_coord'] = sub_traj_df['x_coord']-sub_traj_df['x_coord'].iloc[0]
            sub_traj_df['y_coord'] = sub_traj_df['y_coord']-sub_traj_df['y_coord'].iloc[0]
            sub_traj_df['timestamp'] = sub_traj_df['timestamp']-sub_traj_df['timestamp'].iloc[0]
        
            sub_traj_df['angular_speed'] = numerical_derivative(sub_traj_df['heading'],sub_traj_df['timestamp'],sub_traj_df['delta_timestamps'])
            sub_traj_df['steering_angle']=np.arctan(sub_traj_df['angular_speed']/sub_traj_df['speed']*sub_traj_df['length'])
            
            if abs(sub_traj_df['heading'].max()-sub_traj_df['heading'].min())>1 and not np.isinf(sub_traj_df['angular_speed']).any() and len(sub_traj_df)>30 and sub_traj_df['speed'].min()>5:
                sub_traj_df['trajectory_id'] = traj_count
                traj_count = traj_count+1
                traj=pd.concat((traj,sub_traj_df),axis=0)
                print(len(traj))

        else:
            pass
        # sub_plot[sub_plot['trajectory']==trajectory] = sub_traj_df
    # gdf[gdf['id']==id]=sub_plot
# gdf=gdf[gdf['relevant_trajectory']==1]

print(len(traj))
traj.to_csv('../data/edited_trajectory.csv')


