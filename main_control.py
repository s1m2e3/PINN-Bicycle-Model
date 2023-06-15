import numpy as np
import pandas as pd
import datetime
import utm
import matplotlib.pyplot as plt
import seaborn as sns
from optim_bicycle import *
from pykml import parser
from pykml.factory import write_python_script_for_kml_document

def main():
    # with open('Speedway - Mountain.kml', 'r') as f:
    #     root = parser.parse(f).getroot().Document
    # for pm in root.Placemark:
    #     if "E-W" in pm.name.text:
    #         if "init" in pm.name.text:
    #             x1_init = float(pm.LookAt.longitude.text)
    #             y1_init = float(pm.LookAt.latitude.text)
    #             utm_coord = utm.from_latlon(y1_init, x1_init)
    #             x1_init = utm_coord[0] 
    #             y1_init = utm_coord[1]
    #         elif "final" in pm.name.text:
    #             x1_final = float(pm.LookAt.longitude.text)
    #             y1_final = float(pm.LookAt.latitude.text)
    #             utm_coord = utm.from_latlon(y1_final, x1_final)
    #             x1_final = utm_coord[0] 
    #             y1_final = utm_coord[1]
                
    #     elif "N-S" in pm.name.text:
    #         if "init" in pm.name.text:
    #             x2_init = float(pm.LookAt.longitude.text)
    #             y2_init = float(pm.LookAt.latitude.text)
    #             utm_coord = utm.from_latlon(y2_init, x2_init)
    #             x2_init = utm_coord[0] 
    #             y2_init = utm_coord[1]
    #         elif "final" in pm.name.text:
    #             x2_final = float(pm.LookAt.longitude.text)
    #             y2_final = float(pm.LookAt.latitude.text)
    #             utm_coord = utm.from_latlon(y2_final, x2_final)
    #             x2_final = utm_coord[0] 
    #             y2_final = utm_coord[1]
    df = pd.read_csv("left_turn_through.csv")
    df["utm x"]=df["utm x"]-504000
    df["utm y"]=df["utm y"]-3566000
    df_1 = np.array(df[df["path"]=="through"][["utm x","utm y","heading","vx","vy","ax","ay","relative_propotion"]])
    df_2 = np.array(df[df["path"]=="left turning"][["utm x","utm y","heading","vx","vy","ax","ay","relative_proportion"]])
    
    
    # x1_init = 125
    # y1_init = 618
    # x1_final = 59
    # y1_final = 618
    # x2_init  = 92
    # y2_init  = 646
    # x2_final = 92
    # y2_final = 571
    # theta1_init = np.pi
    # theta1_final = np.pi
    # theta2_init = 3*np.pi/4
    # theta2_final = 3*np.pi/4
    
    # v1x_init = -10
    # v1y_init = 2
    # a1x_init = -2
    # a1y_init = 0
    # v1x_final = 0
    # v1y_final = 0
    # a1x_final = 0
    # a1y_final = 0
    
    # v2x_init = 2
    # v2y_init = -10
    # a2x_init = 0
    # a2y_init = -2
    # v2x_final = 0
    # v2y_final = 0
    # a2x_final = 0
    # a2y_final = 0
   
    car1=df_1
    car2=df_2
    print(car2)
    conflict_x = 504067
    conflict_y = 3566590
    # car1 = np.array([[x1_init,y1_init,theta1_init,v1x_init,v1y_init,a1x_init,a1y_init],\
    #       [x1_final,y1_final,theta1_final,v1x_final,v1y_final,a1x_final,a1y_final]])
    # car2 = np.array([[x2_init,y2_init,theta2_init,v2x_init,v2y_init,a2x_init,a2y_init],\
    #       [x2_final,y2_final,theta2_final,v2x_final,v2y_final,a2x_final,a2y_final]])
    
    last = 8
    time = np.arange(0,last,0.1)
    n_nodes = int(last*10/2)
    n_iterations = 1e0-1
    # for k in range(1):

    states = 4
    co_states = 6
    output_size = states+co_states
    model = XTFC_veh(n_nodes,1,output_size,length=0)
    model.train(n_iterations=n_iterations,x_train=time,y_train_1=car1,y_train_2=car2)
    predictions = model.pred()
    car1 = predictions[:,0:10].detach().numpy()
    car2 = predictions[:,10:20].detach().numpy()
    xcar1= car1[:,0]
    xcar2= car2[:,0]
    ycar1= car1[:,1]
    ycar2= car2[:,1]
    vxcar1= car1[:,4]
    vxcar2= car2[:,4]
    vycar1= car1[:,5]
    vycar2= car2[:,5]
    axcar1= car1[:,8]
    axcar2= car2[:,8]
    aycar1= car1[:,9]
    aycar2= car2[:,9]
    time_string = np.array([str(np.round(tim,1)) for tim in time ])
    plt.figure()
    plt.scatter(xcar1,ycar1)
    for i in range(len(xcar1)):
        if i % 5 ==0:
            plt.text(x=xcar1[i],y=ycar1[i],color="black",s=time_string[i])
    plt.scatter(xcar2,ycar2)
    for i in range(len(xcar2)):
        if i % 5 ==0:
            plt.text(x=xcar2[i],y=ycar2[i],color="red",s=time_string[i])
    plt.title("vehicles in x and y")
    plt.savefig("xycoordinates.png")
    plt.figure(figsize=(10,5))
# 
    plt.scatter(time,(abs(xcar1-conflict_x)+abs(ycar2-conflict_y)))
    plt.hlines(y=7.5,xmin=time[0],xmax=time[-1],colors="red")
    plt.title("Manhattan Distance between cars and conflict point")
    plt.savefig("distance.png")
    # 
# 
    # plt.figure()
    # plt.scatter(time,(abs(xcar1-xcar2)))
    # plt.hlines(y=3,xmin=time[0],xmax=time[-1],colors="red")
    # plt.title("Distance in x between cars over time")
    # plt.savefig("xycoordinates.png")
# 
    plt.figure(figsize=(10,5))
    plt.scatter(time,vxcar1)
    plt.scatter(time,vxcar2)
    plt.title("x speed over time")
    plt.savefig("xspeed.png")
    plt.figure(figsize=(10,5))
    plt.scatter(time,vycar1)
    plt.scatter(time,vycar2)
    plt.title("y speed over time")
    plt.savefig("yspeed.png")
    plt.figure(figsize=(10,5))
    plt.scatter(time,axcar1)
    plt.scatter(time,axcar2)
    plt.title("x accel over time")
    plt.savefig("xaccel.png")
    plt.figure(figsize=(10,5))
    plt.scatter(time,aycar1)
    plt.scatter(time,aycar2)
    plt.title("y accel over time")
    plt.savefig("yaccel.png")
# 

if __name__=="__main__":
    main()
    