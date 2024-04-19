import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
df = pd.read_csv('./data/edited.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
df['timestamp_date']=pd.to_datetime(df['timestamp'],unit='s').dt.minute
unique_ids=df['id'].unique()
condition = (df['id']==unique_ids[1])|(df['id']==unique_ids[2])|(df['id']==unique_ids[3])|(df['id']==unique_ids[0])
sub_plot = df[condition]
gdf = gpd.GeoDataFrame(sub_plot, geometry=gpd.points_from_xy(sub_plot.long, sub_plot.lat))
gdf.crs = "EPSG:4326"
gdf = gdf.to_crs(epsg=3086)
ax = gdf.plot(figsize=(10,10),column='id',categorical=True,alpha=0.5,legend=True,legend_kwds={'loc': 'lower right','title':'Vehicle IDs'})
ax.set_xlabel('X Coordinate',fontsize=16,fontweight='bold')
ax.set_ylabel('Y Coordinate',fontsize=16,fontweight='bold')
ax.set_title('Tampa Florida BSM Data',fontsize=16,fontweight='bold')
ctx.add_basemap(ax,crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
plt.savefig('./trajectory_prediction/images/vehicles.png')
