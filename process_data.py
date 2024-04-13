import pandas as pd
import geopandas as gpd
import numpy as np 
from shapely.geometry import LineString
import matplotlib.pyplot as plt

df = pd.read_csv('./data/edited.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
df['timestamp_date']=pd.to_datetime(df['timestamp'],unit='s').dt.minute
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat))
gdf.crs = "EPSG:4326"
gdf = gdf.to_crs(epsg=3086)
gdf['x_coord'] = gdf['geometry'].x
gdf['y_coord'] = gdf['geometry'].y
unique_ids = np.unique(gdf['id'])
gdf['trajectory']=0
for unique_id in unique_ids:
    sub_df = gdf[gdf['id']==unique_id]
    for i in range(len(sub_df)-1):
        if sub_df['timestamp'].iloc[i+1]-sub_df['timestamp'].iloc[i]>2:
            sub_df['trajectory'].iloc[i+1]=sub_df['trajectory'].iloc[i]+1
        else:
            sub_df['trajectory'].iloc[i+1]=sub_df['trajectory'].iloc[i]
    gdf[gdf['id']==unique_id]=sub_df

for id in unique_ids:
    sub_plot = gdf[gdf['id']==id]
    lines = []
    traj = []
    min_time = []
    max_time = []
    for trajectory in sub_plot['trajectory'].unique():
        traj.append(trajectory)
        sub_traj_df = sub_plot[sub_plot['trajectory']==trajectory]
        min_time.append(min(sub_traj_df['timestamp']))
        max_time.append(max(sub_traj_df['timestamp']))
        sub_traj_df['x_coord'] = sub_traj_df['x_coord']-min(sub_traj_df['x_coord'])
        sub_traj_df['y_coord'] = sub_traj_df['y_coord']-min(sub_traj_df['y_coord'])
        sub_traj_df['heading'] = sub_traj_df['heading']-sub_traj_df['heading'].iloc[0]
        sub_traj_df['timestamp'] = sub_traj_df['timestamp']-min(sub_traj_df['timestamp'])
        sub_plot[sub_plot['trajectory']==trajectory] = sub_traj_df
    gdf[gdf['id']==id]=sub_plot
gdf.to_csv('./data/edited_trajectory.csv')

