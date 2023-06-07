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
    
    x1_init = 506125
    y1_init = 3566618
    x1_final = 506059
    y1_final = 3566618
    x2_init  = 506092
    y2_init  = 3566646
    x2_final = 506092
    y2_final = 3566571
    theta1_init = np.pi
    theta1_final = np.pi
    theta2_init = 3*np.pi/4
    theta2_final = 3*np.pi/4
    
    v1x_init = -10
    v1y_init = 0
    a1x_init = -2
    a1y_init = 0
    v1x_final = 0
    v1y_final = 0
    a1x_final = 0
    a1y_final = 0
    
    v2x_init = 0
    v2y_init = -10
    a2x_init = 0
    a2y_init = -2
    v2x_final = 0
    v2y_final = 0
    a2x_final = 0
    a2y_final = 0
    

    car1 = np.array([[x1_init,y1_init,theta1_init,v1x_init,v1y_init,a1x_init,a1y_init],\
          [x1_final,y1_final,theta1_final,v1x_final,v1y_final,a1x_final,a1y_final]])
    car2 = np.array([[x2_init,y2_init,theta2_init,v2x_init,v2y_init,a2x_init,a2y_init],\
          [x2_final,y2_final,theta2_final,v2x_final,v2y_final,a2x_final,a2y_final]])
    time = np.arange(1,10,0.1)
    
    n_nodes = 30
    n_iterations = 1e2

    states = 4
    co_states = 6
    output_size = states+co_states
    model = XTFC_veh(n_nodes,1,output_size,length=0)
    model.train(n_iterations=n_iterations,x_train=time,y_train_1=car1,y_train_2=car2)

if __name__=="__main__":
    main()
    