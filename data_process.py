import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_lstm = np.loadtxt("prediction_lstm.txt")
df_test_lstm = np.loadtxt("test_lstm.txt")
df_nn = np.loadtxt("prediction_nn.txt")
df_test_nn = np.loadtxt("test_nn.txt")
df_input_lstm = np.loadtxt("input_lstm.txt")

df_nn_resample = np.zeros((911,6))
df_nn_test_resample = np.zeros((911,6))
df_lstm_resample = np.zeros((911,6))
df_lstm_test_resample = np.zeros((911,6))

for i in range(911):

    df_nn_resample[i,:] = df_nn[i*10+1,:]
    df_nn_test_resample[i,:] = df_test_nn[i*10+1,:]
    df_lstm_resample[i,:] = df_lstm[i*10+1,:]
    df_lstm_test_resample[i,:] = df_test_lstm[i*10+1,:]



"""""
# plt.figure(figsize=(20,10))
# plt.scatter(df_nn[0:10,0],df_nn[0:10,1],alpha=0.7)
# plt.scatter(df_test_nn[0:10,0],df_test_nn[0:10,1],alpha=0.7)
# plt.title("X and Y Coordinates Alignment Feedforward Neural Network 1 Second Stream")
# plt.legend(["Prediction","Ground Truth"])
# plt.xlabel("UTM X Coordinate")
# plt.ylabel("UTM Y Coordinate")
# plt.savefig("X and Y Alignment ff.png")


# plt.figure(figsize=(20,10))
# plt.plot(df_nn[0:10,0],alpha=0.7)
# plt.plot(df_test_nn[0:10,0],alpha=0.7)
# plt.title("X Coordinates Alignment Feedforward Neural Network 1 Second Stream")
# plt.legend(["Ground Truth","Prediction"])
# plt.xlabel("Point in Sequence")
# plt.ylabel("UTM X Coordinate")
# plt.savefig("X Coordinate ff.png")



# plt.figure(figsize=(20,10))
# plt.plot(df_nn[0:10,1],alpha=0.7)
# plt.plot(df_test_nn[0:10,1],alpha=0.7)
# plt.title("Y Coordinates Alignment Feedforward Neural Network 1 Second Stream")
# plt.legend(["Ground Truth","Prediction"])
# plt.xlabel("Point in Sequence")
# plt.ylabel("UTM Y Coordinate")
# plt.savefig("Y Coordinate ff.png")


# plt.figure(figsize=(20,10))
# plt.plot(df_nn[0:10,2],alpha=0.7)
# plt.plot(df_test_nn[0:10,2],alpha=0.7)
# plt.title("Heading Angle Feedforward Neural Network 1 Second Stream")
# plt.legend(["Ground Truth","Prediction"])
# plt.xlabel("Point in Sequence")
# plt.ylabel("Angle in Radians")
# plt.savefig("heading ff.png")



# plt.figure(figsize=(20,10))
# plt.scatter(df_nn_resample[:,0],df_nn_resample[:,1],alpha=0.7)
# plt.scatter(df_nn_test_resample[:,0],df_nn_test_resample[:,1],alpha=0.7)
# plt.title("X and Y Coordinates Alignment Feedforward Neural Network Total Broadcast")
# plt.legend(["Prediction","Ground Truth"])
# plt.xlabel("UTM X Coordinate")
# plt.ylabel("UTM Y Coordinate")
# plt.savefig("X and Y Alignment ff_total.png")


# plt.figure(figsize=(20,10))
# plt.plot(df_nn_resample[:,0],alpha=0.7)
# plt.plot(df_nn_test_resample[:,0],alpha=0.7)
# plt.title("X Coordinates Alignment Feedforward Neural Network Total Broadcast")
# plt.legend(["Ground Truth","Prediction"])
# plt.xlabel("Point in Sequence")
# plt.ylabel("UTM X Coordinate")
# plt.savefig("X Coordinate ff_total.png")



# plt.figure(figsize=(20,10))
# plt.plot(df_nn_resample[:,1],alpha=0.7)
# plt.plot(df_nn_test_resample[:,1],alpha=0.7)
# plt.title("Y Coordinates Alignment Feedforward Neural Network Total Broadcast")
# plt.legend(["Ground Truth","Prediction"])
# plt.xlabel("Point in Sequence")
# plt.ylabel("UTM Y Coordinate")
# plt.savefig("Y Coordinate ff_total.png")


# plt.figure(figsize=(20,10))
# plt.plot(df_nn_resample[:,2],alpha=0.7)
# plt.plot(df_nn_test_resample[:,2],alpha=0.7)
# plt.title("Heading Angle Feedforward Neural Network Total Broadcast")
# plt.legend(["Ground Truth","Prediction"])
# plt.xlabel("Point in Sequence")
# plt.ylabel("Angle in Radians")
# plt.savefig("heading ff_total.png")


"""

plt.figure(figsize=(20,10))
plt.scatter(df_lstm[0:10,0],df_lstm[0:10,1],alpha=0.7)
plt.scatter(df_test_nn[0:10,0],df_test_nn[0:10,1],alpha=0.7)
plt.title("X and Y Coordinates Alignment LSTM 1 Second Stream")
plt.legend(["Prediction","Ground Truth"])
plt.xlabel("UTM X Coordinate")
plt.ylabel("UTM Y Coordinate")
plt.savefig("X and Y Alignment lstm.png")


plt.figure(figsize=(20,10))
plt.plot(df_lstm[0:10,0],alpha=0.7)
plt.plot(df_test_nn[0:10,0],alpha=0.7)
plt.title("X Coordinates Alignment LSTM 1 Second Stream")
plt.legend(["Ground Truth","Prediction"])
plt.xlabel("Point in Sequence")
plt.ylabel("UTM X Coordinate")
plt.savefig("X Coordinate lstm.png")



plt.figure(figsize=(20,10))
plt.plot(df_lstm[0:10,1],alpha=0.7)
plt.plot(df_test_nn[0:10,1],alpha=0.7)
plt.title("Y Coordinates Alignment LSTM 1 Second Stream")
plt.legend(["Ground Truth","Prediction"])
plt.xlabel("Point in Sequence")
plt.ylabel("UTM Y Coordinate")
plt.savefig("Y Coordinate lstm.png")


plt.figure(figsize=(20,10))
plt.plot(df_lstm[0:10,2],alpha=0.7)
plt.plot(df_test_nn[0:10,2],alpha=0.7)
plt.title("Heading Angle LSTM 1 Second Stream")
plt.legend(["Ground Truth","Prediction"])
plt.xlabel("Point in Sequence")
plt.ylabel("Angle in Radians")
plt.savefig("heading lstm.png")



plt.figure(figsize=(20,10))
plt.scatter(df_lstm_resample[:,0],df_lstm_resample[:,1],alpha=0.7)
plt.scatter(df_nn_test_resample[:,0],df_nn_test_resample[:,1],alpha=0.7)
plt.title("X and Y Coordinates Alignment LSTM Total Broadcast")
plt.legend(["Prediction","Ground Truth"])
plt.xlabel("UTM X Coordinate")
plt.ylabel("UTM Y Coordinate")
plt.savefig("X and Y Alignment lstm_total.png")


plt.figure(figsize=(20,10))
plt.plot(df_lstm_resample[:,0],alpha=0.7)
plt.plot(df_nn_test_resample[:,0],alpha=0.7)
plt.title("X Coordinates Alignment LSTM Total Broadcast")
plt.legend(["Ground Truth","Prediction"])
plt.xlabel("Point in Sequence")
plt.ylabel("UTM X Coordinate")
plt.savefig("X Coordinate lstm_total.png")



plt.figure(figsize=(20,10))
plt.plot(df_lstm_resample[:,1],alpha=0.7)
plt.plot(df_nn_test_resample[:,1],alpha=0.7)
plt.title("Y Coordinates Alignment LSTM Total Broadcast")
plt.legend(["Ground Truth","Prediction"])
plt.xlabel("Point in Sequence")
plt.ylabel("UTM Y Coordinate")
plt.savefig("Y Coordinate lstm_total.png")


plt.figure(figsize=(20,10))
plt.plot(df_lstm_resample[:,2],alpha=0.7)
plt.plot(df_nn_test_resample[:,2],alpha=0.7)
plt.title("Heading Angle LSTM Total Broadcast")
plt.legend(["Ground Truth","Prediction"])
plt.xlabel("Point in Sequence")
plt.ylabel("Angle in Radians")
plt.savefig("heading lstm_total.png")


