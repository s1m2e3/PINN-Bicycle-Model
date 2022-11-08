import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from bicycle_PINN import bicycle_PINN
import datetime
import utm
from process import prep_df_reg
from predict_ode import *


def main():
    df = pd.read_csv("edited.csv")
    test_df=df[df['sub_group']==df['sub_group'].unique()[2]].reset_index(drop=True)
    #states_reg = test_ode_reg(df_reg)
    #states_lin =test_ode_reg(df_lin)
    pinn_reg = bicycle_PINN(test_df,"reg")
    #pinn_reg = bicycle_PINN(df_reg,"reg")


if __name__=="__main__":
    main()
    