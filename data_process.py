import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

original = pd.read_csv("chosen_data.csv")
#original = original[[]]
df_nn = pd.read_csv("prediction_nn_scaled.csv")
names = list(df_nn.columns)
df_nn=df_nn[names[1:]]
original = original[names[1:]]
#print(original)
#print(df_nn)

df_lstm = pd.read_csv("prediction_lstm_scaled.txt")
df_nn=(df_nn*(original[["x","y","heading","steering_angle","speed","steering_angle_rate"]].max()\
        -original[["x","y","heading","steering_angle","speed","steering_angle_rate"]].min())-\
            -original[["x","y","heading","steering_angle","speed","steering_angle_rate"]].min())
origin_plot = original.loc[int(len(original)*7/10)+1:].reset_index(drop=True)
# plt.figure()
# plt.scatter(origin_plot["x"].iloc[1:51],origin_plot["y"].iloc[1:51],s=5,alpha=0.4)
# plt.scatter(df_nn["x"].iloc[0:50],df_nn["y"].iloc[0:50],s=1,alpha=0.4)
# plt.show()


sns.set_theme()
plt.figure(figsize=(20,10))
plt.scatter(origin_plot["x"].iloc[0:20],origin_plot["y"].iloc[0:20],alpha=0.7)
plt.scatter(df_nn["x"].iloc[0:20],df_nn["y"].iloc[0:20],alpha=0.7)
plt.title("X and Y Coordinates Alignment Feedforward Neural Network")
plt.legend(["Ground Truth","Prediction"])
plt.xlabel("UTM X Coordinate")
plt.ylabel("UTM Y Coordinate")
plt.savefig("X and Y Alignment ff.png")




plt.figure(figsize=(20,10))
plt.plot(origin_plot["x"].iloc[0:20],alpha=0.9)
plt.plot(df_nn["x"].iloc[0:20],alpha=0.9)
plt.title("X Coordinates Alignment Feedforward Neural Network")
plt.legend(["Ground Truth","Prediction"])
plt.xlabel("Point in Sequence")
plt.ylabel("UTM X Coordinate")
plt.savefig("X Coordinate ff.png")


plt.figure(figsize=(20,10))
plt.plot(origin_plot["y"].iloc[0:20],alpha=0.9)
plt.plot(df_nn["y"].iloc[0:20],alpha=0.9)
plt.title("Y Coordinates Alignment Feedforward Neural Network")
plt.legend(["Ground Truth","Prediction"])
plt.xlabel("Point in Sequence")
plt.ylabel("UTM Y Coordinate")
plt.savefig("Y Coordinate ff.png")



plt.figure(figsize=(20,10))
plt.plot(origin_plot["heading"].iloc[0:20],alpha=0.9)
plt.plot(df_nn["heading"].iloc[0:20],alpha=0.9)
plt.title("Heading Feedforward Neural Network")
plt.legend(["Ground Truth","Prediction"])
plt.xlabel("Point in Sequence")
plt.ylabel("Angle in rad")
plt.savefig("angle ff.png")