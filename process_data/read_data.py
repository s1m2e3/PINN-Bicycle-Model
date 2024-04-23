import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import json
import matplotlib.pyplot as plt 
import contextily as ctx



data = pd.read_csv("../data/Tampa_CV_Pilot_Basic_Safety_Message__BSM__Sample.csv")
data = data[['metadata_generatedAt','coreData_id','coreData_position_lat','coreData_position_long','coreData_angle','coreData_speed','coreData_heading','coreData_size','metadata_generatedAt_timeOfDay','coreData_secMark']]
data['timestamp'] = pd.to_datetime(data['metadata_generatedAt'],unit='ns')
data['timestamp']=data['timestamp'].astype(int)/10**9
data['timestamp'] = data['timestamp']+data['coreData_secMark']%1000*0.001
data['coreData_size'] = data['coreData_size'].apply(lambda x: json.loads(x))
data['width']= data['coreData_size'].apply(lambda x: x['width']).apply(lambda x: int(x))
data['length']= data['coreData_size'].apply(lambda x: x['length']).apply(lambda x: int(x))
data.drop('coreData_size', axis=1, inplace=True)
data.drop('metadata_generatedAt_timeOfDay',axis=1,inplace=True)
data.drop('metadata_generatedAt',axis=1,inplace=True)
data.drop('coreData_secMark',axis=1,inplace=True)
data.columns = ['id', 'lat', 'long', 'angle', 'speed', 'heading','timestamp', 'width', 'length']
data[['lat', 'long']] = data[['lat', 'long']].astype(float)/1e7
data['speed']=data['speed'].astype(float)*0.02
data['heading']=data['heading'].astype(float)*0.0125
data['angle']=data['angle'].astype(float)*1.5

data.sort_values(by=['timestamp','id'],inplace=True)
data = data.reset_index(drop=True)
data.to_csv('../data/edited.csv')





