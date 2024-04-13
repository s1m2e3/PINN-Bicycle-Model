from trajectory_prediction.src.bicycle_PINN import *
import matplotlib.pyplot as plt

class XTFC_veh(PIELM):
    def __init__(self,n_nodes,input_size,output_size,length,low_w=-1,high_w=1,low_b=-1,high_b=1,activation_function="tanh",d=0.5,k=1):
        super().__init__(n_nodes,input_size,output_size,length,low_w=-1,high_w=1,low_b=-1,high_b=1,activation_function="tanh")

        self.length= length
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nodes = n_nodes
        self.W = (torch.randn(size=(n_nodes,1),dtype=torch.float)*(high_w-low_w)+low_w)
        self.b = (torch.randn(size=(n_nodes,1),dtype=torch.float)*(high_b-low_b)+low_b)
        
        self.betas = torch.ones(size=(2*output_size*n_nodes,),requires_grad=True,dtype=torch.float)
        #betas 0:n_nodes = pred_x_1
        #betas n_nodes:n_nodes*2 = pred_y_1
        #betas n_nodes*2:n_nodes*3 = pred_theta_1
        #betas n_nodes*3:n_nodes*4 = pred_delta_1
        #betas n_nodes*4:n_nodes*5 = pred_lambda_x_1
        #betas n_nodes*5:n_nodes*6 = pred_lambda_y_1
        #betas n_nodes*6:n_nodes*7 = pred_lambda_vx_1
        #betas n_nodes*7:n_nodes*8 = pred_lambda_vy_1
        #betas n_nodes*8:n_nodes*9 = pred_lambda_theta_1
        #betas n_nodes*9:n_nodes*10 = pred_lambda_delta_1

        #betas n_nodes*10:n_nodes*11 = pred_x_2
        #betas n_nodes*11:n_nodes*12 = pred_y_2
        #betas n_nodes*12:n_nodes*13 = pred_theta_2
        #betas n_nodes*13:n_nodes*14 = pred_delta_2
        #betas n_nodes*14:n_nodes*15 = pred_lambda_x_2
        #betas n_nodes*15:n_nodes*16 = pred_lambda_y_2
        #betas n_nodes*16:n_nodes*17 = pred_lambda_vx_2
        #betas n_nodes*17:n_nodes*18 = pred_lambda_vy_2
        #betas n_nodes*18:n_nodes*19 = pred_lambda_theta_2
        #betas n_nodes*19:n_nodes*20 = pred_lambda_delta_2
        self.d = d
        self.k = k

    def predict_jacobian(self,betas):
        
              
        h = self.get_h(self.x_train)
        dh = self.get_dh(self.x_train)
        dhh = self.get_dhh(self.x_train)

        bx_1 = betas[0:self.nodes]
        by_1 = betas[self.nodes:self.nodes*2]
        btheta_1 = betas[self.nodes*2:self.nodes*3]
        bdelta_1 = betas[self.nodes*3:self.nodes*4]
        blambda_x_1 = betas[self.nodes*4:self.nodes*5]
        blambda_y_1 = betas[self.nodes*5:self.nodes*6]
        blambda_dx_1 = betas[self.nodes*6:self.nodes*7]
        blambda_dy_1 = betas[self.nodes*7:self.nodes*8]
        blambda_dtheta_1 = betas[self.nodes*8:self.nodes*9]
        blambda_ddelta_1 = betas[self.nodes*9:self.nodes*10]
        
        bx_2 = betas[self.nodes*10:self.nodes*11]
        by_2 = betas[self.nodes*11:self.nodes*12]
        btheta_2 = betas[self.nodes*12:self.nodes*13]
        bdelta_2 = betas[self.nodes*13:self.nodes*14]
        blambda_x_2 = betas[self.nodes*14:self.nodes*15]
        blambda_y_2 = betas[self.nodes*15:self.nodes*16]
        blambda_dx_2 = betas[self.nodes*16:self.nodes*17]
        blambda_dy_2 = betas[self.nodes*17:self.nodes*18]
        blambda_dtheta_2 = betas[self.nodes*18:self.nodes*19]
        blambda_ddelta_2 = betas[self.nodes*19:self.nodes*20]

        init_x_1 = self.y_train_1[0,0]        
        final_x_1 = self.y_train_1[-1,0]
        init_y_1 = self.y_train_1[0,1]
        final_y_1 = self.y_train_1[-1,1]
        init_theta_1 = self.y_train_1[0,2]        
        final_theta_1 = self.y_train_1[-1,2]
        init_vx_1 = self.y_train_1[0,3]        
        init_vy_1 = self.y_train_1[0,4]
        init_ax_1 = self.y_train_1[0,5]        
        init_ay_1 = self.y_train_1[0,6]
        init_h_1 = self.get_h(self.x_train[0])
        final_h_1 = self.get_h(self.x_train[-1])
        init_dh_1 = self.get_dh(self.x_train[0])
        init_dhh_1 = self.get_dhh(self.x_train[0])
        

        init_x_2 = self.y_train_2[0,0]        
        final_x_2 = self.y_train_2[-1,0]
        init_y_2 = self.y_train_2[0,1]
        final_y_2 = self.y_train_2[-1,1]
        init_theta_2 = self.y_train_2[0,2]        
        final_theta_2 = self.y_train_2[-1,2]
        init_vx_2 = self.y_train_2[0,3]        
        init_vy_2 = self.y_train_2[0,4]
        init_ax_2 = self.y_train_2[0,5]        
        init_ay_2 = self.y_train_2[0,6]

        cross_x_2 = self.y_train_2[1,0]
        cross_y_2 = self.y_train_2[1,1]
        r1_x_2 =self.y_train_2[2,0]
        r1_y_2 =self.y_train_2[2,1]
        r2_x_2 =self.y_train_2[3,0]
        r2_y_2 =self.y_train_2[3,1]
        r3_x_2 =self.y_train_2[4,0]
        r3_y_2 =self.y_train_2[4,1]

        rel_cross =  self.y_train_2[1,-1]
        rel_r1 = self.y_train_2[2,-1]
        rel_r2 = self.y_train_2[3,-1]
        rel_r3 = self.y_train_2[4,-1]

        final_time = self.x_train[-1].numpy()[0]
        init_time = self.x_train[0].numpy()[0]
        
        
        cross_time= ((final_time-init_time)*rel_cross+init_time).reshape(self.x_train[0].shape)
        r1_time= ((final_time-init_time)*rel_r1+init_time).reshape(self.x_train[0].shape)
        r2_time= ((final_time-init_time)*rel_r2+init_time).reshape(self.x_train[0].shape)
        r3_time= ((final_time-init_time)*rel_r3+init_time).reshape(self.x_train[0].shape)
        
        cross_index = np.where(self.x_train>=cross_time.numpy()[0])[0][0]
        cross_time = self.x_train[cross_index]
        r1_index = np.where(self.x_train>=r1_time.numpy()[0])[0][0]
        r1_time = self.x_train[r1_index]
        r2_index = np.where(self.x_train>=r2_time.numpy()[0])[0][0]
        r2_time = self.x_train[r2_index]
        r3_index = np.where(self.x_train>=r3_time.numpy()[0])[0][0]
        r3_time = self.x_train[r3_index]
        

        init_h_2 = self.get_h(self.x_train[0])
        cross_h_2 = self.get_h(cross_time)
        r1_h_2 = self.get_h(r1_time)
        r2_h_2 = self.get_h(r2_time)
        r3_h_2 = self.get_h(r3_time)
        final_h_2 = self.get_h(self.x_train[-1])
        init_dh_2 = self.get_dh(self.x_train[0])
        init_ddh_2 = self.get_dhh(self.x_train[0])
        cross_time = cross_time.numpy()[0]
        r1_time = r1_time.numpy()[0]
        r2_time = r2_time.numpy()[0]
        r3_time = r3_time.numpy()[0]
        
        
        # x_linspace[0:len(self.x_train)/2]=torch.linspace(init_x_2,final_x_2,int(abs(final_x_2.detach().numpy()-init_x_2.detach().numpy())/2))
        # x_linspace[len(self.x_train)/2:]= final_x_2
        # y_linspace[len(self.x_train)/2:]=torch.linspace(init_y_2,final_y_2,int(abs(final_y_2.detach().numpy()-init_y_2.detach().numpy())/2))
        # y_linspace[0:len(self.x_train)/2]= init_y_2
        # plt.figure()
        # plt.plot(x_linspace.detach().numpy(),plt.plot(y_linspace.detach().numpy()))
        # plt.show()
        """""
        For a 4 point constrained problem on x and y  we have: 

        Consider support functions the polynomials: t^0,t^1,t^2,t^3: Then for x and y we have:
        x(0)       [[1,0,0,0]
        x(f)       [1,t,t^2,t^3]
        xdot(0)    [0,1,0,0]
        xdotdot(0) [0,0,2,0]]

        Consider support functions the polynomials: t^0,t^1,t^2,t^3,t^4,t^5: Then for x and y we have:
        x(0)       [[1,t(0),t(0)^2,t(0)^3,t(0)^4,t(0)^5]
        x(r1)      [1,t(r1),t(r1)^2,t(r1)^3,t(r1)^4,t(r1)^5]
        x(r3)      [1,t(r3),t(r3)^2,t(r3)^3,t(r3)^4,t(r3)^5]
        x(f)       [1,t(rf),t(rf)^2,t(rf)^3]
        xdot(0)    [0,1,0,0]
        xdotdot(0) [0,0,2,0]]
        

        For a 2 point constrained problem on theta we have:
        Consider support functions the polynomials: t^0,t^1: Then for theta have:
        theta(0)       [[1,0]
        theta(f)       [1,t]
        """""
        
        support_function_matrix = np.array([[1,init_time,init_time**2,init_time**3],[1,final_time,final_time**2,final_time**3],[0,1,2*init_time,3*init_time**2],[0,0,2,6*init_time]])
        coefficients_matrix = torch.tensor(np.linalg.inv(support_function_matrix),dtype=torch.float)
        
        free_support_function_matrix = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train,self.x_train**2,self.x_train**3))
        d_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape),2*self.x_train,3*self.x_train**2))
        dd_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.zeros(size=self.x_train.shape),2*torch.ones(size=self.x_train.shape),6*self.x_train))

        
        support_function_matrix_2 = np.array([[1,init_time,init_time**2,init_time**3,init_time**4,init_time**5,init_time**6,init_time**7]\
                                              ,[1,final_time,final_time**2,final_time**3,final_time**4,final_time**5,final_time**6,final_time**7]\
                                              ,[1,cross_time,cross_time**2,cross_time**3,cross_time**4,cross_time**5,cross_time**6,cross_time**7]\
                                              ,[1,r1_time,r1_time**2,r1_time**3,r1_time**4,r1_time**5,r1_time**6,r1_time**7]\
                                              ,[1,r2_time,r2_time**2,r2_time**3,r2_time**4,r2_time**5,r2_time**6,r2_time**7]\
                                              ,[1,r3_time,r3_time**2,r3_time**3,r3_time**4,r3_time**5,r3_time**6,r3_time**7]\
                                              ,[0,1,2*init_time,3*init_time**2,4*init_time**3,5*init_time**4,6*init_time**5,7*init_time**6]\
                                              ,[0,0,2,6*init_time,12*init_time**2,20*init_time**3,30*init_time**4,42*init_time**5]])
        
        coefficients_matrix_2 = torch.tensor(np.linalg.inv(support_function_matrix_2),dtype=torch.float)
        free_support_function_matrix_2 = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train,self.x_train**2,self.x_train**3,self.x_train**4,self.x_train**5,self.x_train**6,self.x_train**7))
        d_free_support_function_matrix_2 = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape),2*self.x_train,3*self.x_train**2\
                                                         ,4*self.x_train**3,5*self.x_train**4,6*self.x_train**5,7*self.x_train**6))
        dd_free_support_function_matrix_2 = torch.hstack((torch.zeros(size=self.x_train.shape),torch.zeros(size=self.x_train.shape),2*torch.ones(size=self.x_train.shape)\
                                                          ,6*self.x_train,12*self.x_train**2,20*self.x_train**3,30*self.x_train**4,42*self.x_train**5))

        
        phis = torch.matmul(free_support_function_matrix,coefficients_matrix)
        phi1 = phis[:,0].reshape(len(self.x_train),1)
        phi2 = phis[:,1].reshape(len(self.x_train),1)
        phi3 = phis[:,2].reshape(len(self.x_train),1)
        phi4 = phis[:,3].reshape(len(self.x_train),1)
        d_phis = torch.matmul(d_free_support_function_matrix,coefficients_matrix)
        d_phi1 = d_phis[:,0].reshape(len(self.x_train),1)
        d_phi2 = d_phis[:,1].reshape(len(self.x_train),1)
        d_phi3 = d_phis[:,2].reshape(len(self.x_train),1)
        d_phi4 = d_phis[:,3].reshape(len(self.x_train),1)
        dd_phis = torch.matmul(dd_free_support_function_matrix,coefficients_matrix)
        dd_phi1 = dd_phis[:,0].reshape(len(self.x_train),1)
        dd_phi2 = dd_phis[:,1].reshape(len(self.x_train),1)
        dd_phi3 = dd_phis[:,2].reshape(len(self.x_train),1)
        dd_phi4 = dd_phis[:,3].reshape(len(self.x_train),1)

        
        phi1_h1 = torch.matmul(-phi1,init_h_1)
        phi1_x1_init = phi1*init_x_1
        phi1_y1_init = phi1*init_y_1
        phi2_hf = torch.matmul(-phi2,final_h_1)
        phi2_x1_final = phi2*final_x_1
        phi2_y1_final = phi2*final_y_1
        phi3_dh1 = torch.matmul(-phi3,init_dh_1)
        phi3_vx1_init = phi3*init_vx_1
        phi3_vy1_init = phi3*init_vy_1
        phi4_ddh1 = torch.matmul(-phi4,init_dhh_1)
        phi4_ax1_init = phi4*init_ax_1
        phi4_ay1_init = phi4*init_ay_1
        
        d_phi1_h1 = torch.matmul(-d_phi1,init_h_1)
        d_phi1_x1_init = d_phi1*init_x_1
        d_phi1_y1_init = d_phi1*init_y_1
        d_phi2_hf = torch.matmul(-d_phi2,final_h_1)
        d_phi2_x1_final = d_phi2*final_x_1
        d_phi2_y1_final = d_phi2*final_y_1
        d_phi3_dh1 = torch.matmul(-d_phi3,init_dh_1)
        d_phi3_vx1_init = d_phi3*init_vx_1
        d_phi3_vy1_init = d_phi3*init_vy_1
        d_phi4_ddh1 = torch.matmul(-d_phi4,init_dhh_1)
        d_phi4_ax1_init = d_phi4*init_ax_1
        d_phi4_ay1_init = d_phi4*init_ay_1
        
        dd_phi1_h1 = torch.matmul(-dd_phi1,init_h_1)
        dd_phi1_x1_init = dd_phi1*init_x_1
        dd_phi1_y1_init = dd_phi1*init_y_1
        dd_phi2_hf = torch.matmul(-dd_phi2,final_h_1)
        dd_phi2_x1_final = dd_phi2*final_x_1
        dd_phi2_y1_final = dd_phi2*final_y_1
        dd_phi3_dh1 = torch.matmul(-dd_phi3,init_dh_1)
        dd_phi3_vx1_init = dd_phi3*init_vx_1
        dd_phi3_vy1_init = dd_phi3*init_vy_1
        dd_phi4_ddh1 = torch.matmul(-dd_phi4,init_dhh_1)
        dd_phi4_ax1_init = dd_phi4*init_ax_1
        dd_phi4_ay1_init = dd_phi4*init_ay_1
        
   

        phis_2 = torch.matmul(free_support_function_matrix_2,coefficients_matrix_2)
        phi1_2 = phis_2[:,0].reshape(len(self.x_train),1)
        phi2_2 = phis_2[:,1].reshape(len(self.x_train),1)
        phi3_2 = phis_2[:,2].reshape(len(self.x_train),1)
        phi4_2 = phis_2[:,3].reshape(len(self.x_train),1)
        phi5_2 = phis_2[:,4].reshape(len(self.x_train),1)
        phi6_2 = phis_2[:,5].reshape(len(self.x_train),1)
        phi7_2 = phis_2[:,6].reshape(len(self.x_train),1)
        phi8_2 = phis_2[:,7].reshape(len(self.x_train),1)

        d_phis_2 = torch.matmul(d_free_support_function_matrix_2,coefficients_matrix_2)
        d_phi1_2 = d_phis_2[:,0].reshape(len(self.x_train),1)
        d_phi2_2 = d_phis_2[:,1].reshape(len(self.x_train),1)
        d_phi3_2 = d_phis_2[:,2].reshape(len(self.x_train),1)
        d_phi4_2 = d_phis_2[:,3].reshape(len(self.x_train),1)
        d_phi5_2 = d_phis_2[:,4].reshape(len(self.x_train),1)
        d_phi6_2 = d_phis_2[:,5].reshape(len(self.x_train),1)
        d_phi7_2 = d_phis_2[:,6].reshape(len(self.x_train),1)
        d_phi8_2 = d_phis_2[:,7].reshape(len(self.x_train),1)
        
        dd_phis_2 = torch.matmul(dd_free_support_function_matrix_2,coefficients_matrix_2)
        dd_phi1_2 = dd_phis_2[:,0].reshape(len(self.x_train),1)
        dd_phi2_2 = dd_phis_2[:,1].reshape(len(self.x_train),1)
        dd_phi3_2 = dd_phis_2[:,2].reshape(len(self.x_train),1)
        dd_phi4_2 = dd_phis_2[:,3].reshape(len(self.x_train),1)
        dd_phi5_2 = dd_phis_2[:,4].reshape(len(self.x_train),1)
        dd_phi6_2 = dd_phis_2[:,5].reshape(len(self.x_train),1)
        dd_phi7_2 = dd_phis_2[:,6].reshape(len(self.x_train),1)
        dd_phi8_2 = dd_phis_2[:,7].reshape(len(self.x_train),1)

        
        phi1_h1_2 = torch.matmul(-phi1_2,init_h_2)
        phi1_x2_init_2 = phi1_2*init_x_2
        phi1_y2_init_2 = phi1_2*init_y_2
        phi2_hf_2 = torch.matmul(-phi2_2,final_h_2)
        phi2_x2_final_2 = phi2_2*final_x_2
        phi2_y2_final_2 = phi2_2*final_y_2
        phi3_hcross_2 = torch.matmul(-phi3_2,cross_h_2)
        phi3_x2_cross_2 = phi3_2*cross_x_2
        phi3_y2_cross_2 = phi3_2*cross_y_2
        phi4_hr1_2 = torch.matmul(-phi4_2,r1_h_2)
        phi4_x2_r1_2 = phi4_2*r1_x_2
        phi4_y2_r1_2 = phi4_2*r1_y_2
        phi5_hr2_2 = torch.matmul(-phi5_2,r2_h_2)
        phi5_x2_r2_2 = phi5_2*r2_x_2
        phi5_y2_r2_2 = phi5_2*r2_y_2
        phi6_hr3_2 = torch.matmul(-phi6_2,r3_h_2)
        phi6_x2_r3_2 = phi6_2*r3_x_2
        phi6_y2_r3_2 = phi6_2*r3_y_2
        phi7_dh2_2 = torch.matmul(-phi7_2,init_dh_2)
        phi7_vx2_init_2 = phi7_2*init_vx_2
        phi7_vy2_init_2 = phi7_2*init_vy_2
        phi8_ddh2_2 = torch.matmul(-phi8_2,init_ddh_2)
        phi8_ax2_init_2 = phi8_2*init_ax_2
        phi8_ay2_init_2 = phi8_2*init_ay_2
        
        d_phi1_h1_2 = torch.matmul(-d_phi1_2,init_h_2)
        d_phi1_x2_init_2 = d_phi1_2*init_x_2
        d_phi1_y2_init_2 = d_phi1_2*init_y_2
        d_phi2_hf_2 = torch.matmul(-d_phi2_2,final_h_2)
        d_phi2_x2_final_2 = d_phi2_2*final_x_2
        d_phi2_y2_final_2 = d_phi2_2*final_y_2
        d_phi3_hcross_2 = torch.matmul(-d_phi3_2,cross_h_2)
        d_phi3_x2_cross_2 = d_phi3_2*cross_x_2
        d_phi3_y2_cross_2 = d_phi3_2*cross_y_2
        d_phi4_hr1_2 = torch.matmul(-d_phi4_2,r1_h_2)
        d_phi4_x2_r1_2 = d_phi4_2*r1_x_2
        d_phi4_y2_r1_2 = d_phi4_2*r1_y_2
        d_phi5_hr2_2 = torch.matmul(-d_phi5_2,r2_h_2)
        d_phi5_x2_r2_2 = d_phi5_2*r2_x_2
        d_phi5_y2_r2_2 = d_phi5_2*r2_y_2
        d_phi6_hr3_2 = torch.matmul(-d_phi6_2,r3_h_2)
        d_phi6_x2_r3_2 = d_phi6_2*r3_x_2
        d_phi6_y2_r3_2 = d_phi6_2*r3_y_2
        d_phi7_dh2_2 = torch.matmul(-d_phi7_2,init_dh_2)
        d_phi7_vx2_init_2 = d_phi7_2*init_vx_2
        d_phi7_vy2_init_2 = d_phi7_2*init_vy_2
        d_phi8_ddh2_2 = torch.matmul(-d_phi8_2,init_ddh_2)
        d_phi8_ax2_init_2 = d_phi8_2*init_ax_2
        d_phi8_ay2_init_2 = d_phi8_2*init_ay_2

        dd_phi1_h1_2 = torch.matmul(-dd_phi1_2,init_h_2)
        dd_phi1_x2_init_2 = dd_phi1_2*init_x_2
        dd_phi1_y2_init_2 = dd_phi1_2*init_y_2
        dd_phi2_hf_2 = torch.matmul(-dd_phi2_2,final_h_2)
        dd_phi2_x2_final_2 = dd_phi2_2*final_x_2
        dd_phi2_y2_final_2 = dd_phi2_2*final_y_2
        dd_phi3_hcross_2 = torch.matmul(-dd_phi3_2,cross_h_2)
        dd_phi3_x2_cross_2 = dd_phi3_2*cross_x_2
        dd_phi3_y2_cross_2 = dd_phi3_2*cross_y_2
        dd_phi4_hr1_2 = torch.matmul(-dd_phi4_2,r1_h_2)
        dd_phi4_x2_r1_2 = dd_phi4_2*r1_x_2
        dd_phi4_y2_r1_2 = dd_phi4_2*r1_y_2
        dd_phi5_hr2_2 = torch.matmul(-dd_phi5_2,r2_h_2)
        dd_phi5_x2_r2_2 = dd_phi5_2*r2_x_2
        dd_phi5_y2_r2_2 = dd_phi5_2*r2_y_2
        dd_phi6_hr3_2 = torch.matmul(-dd_phi6_2,r3_h_2)
        dd_phi6_x2_r3_2 = dd_phi6_2*r3_x_2
        dd_phi6_y2_r3_2 = dd_phi6_2*r3_y_2
        dd_phi7_dh2_2 = torch.matmul(-dd_phi7_2,init_dh_2)
        dd_phi7_vx2_init_2 = dd_phi7_2*init_vx_2
        dd_phi7_vy2_init_2 = dd_phi7_2*init_vy_2
        dd_phi8_ddh2_2 = torch.matmul(-dd_phi8_2,init_ddh_2)
        dd_phi8_ax2_init_2 = dd_phi8_2*init_ax_2
        dd_phi8_ay2_init_2 = dd_phi8_2*init_ay_2



        
        hx_1 = torch.matmul(h.add(phi1_h1).add(phi2_hf).add(phi3_dh1).add(phi4_ddh1),bx_1).reshape(self.x_train.shape)\
            .add(phi1_x1_init).add(phi2_x1_final).add(phi3_vx1_init/self.c).add(phi4_ax1_init/self.c**2)
        
        dhx_1 = self.c*torch.matmul(dh.add(d_phi1_h1).add(d_phi2_hf).add(d_phi3_dh1).add(d_phi4_ddh1),bx_1).reshape(self.x_train.shape)\
            .add(d_phi1_x1_init).add(d_phi2_x1_final).add(d_phi3_vx1_init/self.c).add(d_phi4_ax1_init/self.c**2)
        
        ddhx_1 = self.c**2*torch.matmul(dhh.add(dd_phi1_h1).add(dd_phi2_hf).add(dd_phi3_dh1).add(dd_phi4_ddh1),bx_1).reshape(self.x_train.shape)\
            .add(dd_phi1_x1_init).add(dd_phi2_x1_final).add(dd_phi3_vx1_init/self.c).add(dd_phi4_ax1_init/self.c**2)
        
        hy_1 = torch.matmul(h.add(phi1_h1).add(phi2_hf).add(phi3_dh1).add(phi4_ddh1),by_1).reshape(self.x_train.shape)\
            .add(phi1_y1_init).add(phi2_y1_final).add(phi3_vy1_init/self.c).add(phi4_ay1_init/self.c**2)
        
        dhy_1 = self.c*torch.matmul(dh.add(d_phi1_h1).add(d_phi2_hf).add(d_phi3_dh1).add(d_phi4_ddh1),by_1).reshape(self.x_train.shape)\
            .add(d_phi1_y1_init).add(d_phi2_y1_final).add(d_phi3_vy1_init/self.c).add(d_phi4_ay1_init/self.c**2)
        
        ddhy_1 = self.c**2*torch.matmul(dhh.add(dd_phi1_h1).add(dd_phi2_hf).add(dd_phi3_dh1).add(dd_phi4_ddh1),by_1).reshape(self.x_train.shape)\
            .add(dd_phi1_y1_init).add(dd_phi2_y1_final).add(dd_phi3_vy1_init/self.c).add(dd_phi4_ay1_init/self.c**2)
        
        hx_2 = torch.matmul(h.add(phi1_h1_2).add(phi2_hf_2).add(phi3_hcross_2).add(phi4_hr1_2).add(phi5_hr2_2).add(phi6_hr3_2).add(phi7_dh2_2).add(phi8_ddh2_2),bx_2).reshape(self.x_train.shape)\
            .add(phi1_x2_init_2).add(phi2_x2_final_2).add(phi3_x2_cross_2).add(phi4_x2_r1_2).add(phi5_x2_r2_2).add(phi6_x2_r3_2).add(phi7_vx2_init_2/self.c).add(phi8_ax2_init_2/self.c**2)
        
        dhx_2 = self.c*torch.matmul(dh.add(d_phi1_h1_2).add(d_phi2_hf_2).add(d_phi3_hcross_2).add(d_phi4_hr1_2).add(d_phi5_hr2_2).add(d_phi6_hr3_2).add(d_phi7_dh2_2).add(d_phi8_ddh2_2),bx_2).reshape(self.x_train.shape)\
            .add(d_phi1_x2_init_2).add(d_phi2_x2_final_2).add(d_phi3_x2_cross_2).add(d_phi4_x2_r1_2).add(d_phi5_x2_r2_2).add(d_phi6_x2_r3_2).add(d_phi7_vx2_init_2/self.c).add(d_phi8_ax2_init_2/self.c**2)

        ddhx_2 = self.c**2*torch.matmul(dhh.add(dd_phi1_h1_2).add(dd_phi2_hf_2).add(dd_phi3_hcross_2).add(dd_phi4_hr1_2).add(dd_phi5_hr2_2).add(dd_phi6_hr3_2).add(dd_phi7_dh2_2).add(dd_phi8_ddh2_2),bx_2).reshape(self.x_train.shape)\
            .add(dd_phi1_x2_init_2).add(dd_phi2_x2_final_2).add(dd_phi3_x2_cross_2).add(dd_phi4_x2_r1_2).add(dd_phi5_x2_r2_2).add(dd_phi6_x2_r3_2).add(dd_phi7_vx2_init_2/self.c).add(dd_phi8_ax2_init_2/self.c**2)

        hy_2 = torch.matmul(h.add(phi1_h1_2).add(phi2_hf_2).add(phi3_hcross_2).add(phi4_hr1_2).add(phi5_hr2_2).add(phi6_hr3_2).add(phi7_dh2_2).add(phi8_ddh2_2),by_2).reshape(self.x_train.shape)\
            .add(phi1_y2_init_2).add(phi2_y2_final_2).add(phi3_y2_cross_2).add(phi4_y2_r1_2).add(phi5_y2_r2_2).add(phi6_y2_r3_2).add(phi7_vy2_init_2/self.c).add(phi8_ay2_init_2/self.c**2)
        
        dhy_2 = self.c*torch.matmul(dh.add(d_phi1_h1_2).add(d_phi2_hf_2).add(d_phi3_hcross_2).add(d_phi4_hr1_2).add(d_phi5_hr2_2).add(d_phi6_hr3_2).add(d_phi7_dh2_2).add(d_phi8_ddh2_2),by_2).reshape(self.x_train.shape)\
            .add(d_phi1_y2_init_2).add(d_phi2_y2_final_2).add(d_phi3_y2_cross_2).add(d_phi4_y2_r1_2).add(d_phi5_y2_r2_2).add(d_phi6_y2_r3_2).add(d_phi7_vy2_init_2/self.c).add(d_phi8_ay2_init_2/self.c**2)

        ddhy_2 = self.c**2*torch.matmul(dhh.add(dd_phi1_h1_2).add(dd_phi2_hf_2).add(dd_phi3_hcross_2).add(dd_phi4_hr1_2).add(dd_phi5_hr2_2).add(dd_phi6_hr3_2).add(dd_phi7_dh2_2).add(dd_phi8_ddh2_2),by_2).reshape(self.x_train.shape)\
            .add(dd_phi1_y2_init_2).add(dd_phi2_y2_final_2).add(dd_phi3_y2_cross_2).add(dd_phi4_y2_r1_2).add(dd_phi5_y2_r2_2).add(dd_phi6_y2_r3_2).add(dd_phi7_vy2_init_2/self.c).add(dd_phi8_ay2_init_2/self.c**2)

        y_y_1 = hy_1
        y_dy_1 = dhy_1
        y_ddy_1 = ddhy_1
        y_x_1 = hx_1
        y_dx_1 = dhx_1
        y_ddx_1 = ddhx_1
        y_y_2 = hy_2
        y_dy_2 = dhy_2
        y_ddy_2 = ddhy_2
        y_x_2 = hx_2
        y_dx_2 = dhx_2
        y_ddx_2 = ddhx_2
        heavy = lambda x,k: 1/2 +1/2*torch.tanh(k*x)
        x_c = 68
        y_c = 590.5

        #Add inequalities for x and y for car 1

        f_r_u_1 = init_y_1+self.d
        f_r_l_1 = init_y_1-self.d
        heaviside_r_u = heavy(y_y_1-f_r_u_1,self.k)
        heaviside_r_l =  heavy(f_r_l_1-y_y_1,self.k)

        
        min_dist =  y_x_2-y_x_1+y_y_2-y_y_1
        heaviside_min_dist = heavy(self.d-min_dist,self.k)
        heaviside_r1 = heavy(x_c-y_x_1,self.k)
        heaviside_r2 = heavy(y_y_2-y_c,self.k)
        region1 = heaviside_min_dist*heaviside_r1
        region2 = heaviside_min_dist*heaviside_r2
        
    
        # hx_1 = y_x_1 + (-(self.d+4.9)+y_x_2+y_y_2-y_y_1-y_x_1)*(region1)
        # dhx_1 = y_dx_1 + (y_dx_2+y_dy_2-y_dy_1-y_dx_1)*(region1)
        # ddhx_1 = y_ddx_1 + (y_ddx_2+y_ddy_2-y_ddy_1-y_ddx_1)*(region1)

        # hy_1 = y_y_1 + (-(self.d+4.9)+y_x_2-y_x_1+y_y_2-y_y_1)*(region1)
        # dhy_1 = y_dy_1 + (y_dx_2-y_dx_1+y_dy_2-y_dy_1)*(region1)
        # ddhy_1 = y_ddy_1 + (y_ddx_2-y_ddx_1+y_ddy_2-y_ddy_1)*(region1)
        
       
        hx_2 = y_x_2 + ((self.d+4.9)+y_x_1-y_y_2+y_y_1-y_x_2)*(region2)
        dhx_2 = y_dx_2 + (y_dx_1-y_dy_2+y_dy_1-y_dx_2)*(region2)
        ddhx_2 = y_ddx_2 + (y_ddx_1-y_ddy_2+y_ddy_1-y_ddx_2)*(region2)
      
        hy_2 = y_y_2 + ((self.d+4.9)+y_x_1-y_x_2+y_y_1-y_y_2)*(region2)
        dhy_2 = y_dy_2 + (y_dx_1-y_dx_2+y_dy_1-y_dy_2)*(region2)
        ddhy_2 = y_ddy_2 + (y_dx_1-y_dx_2+y_dy_1-y_dy_2)*(region2)

        y_y_1 = hy_1
        y_dy_1 = dhy_1
        y_ddy_1 = ddhy_1


        hy_1 = y_y_1 +(f_r_u_1-y_y_1)*(heaviside_r_u)+(f_r_l_1-y_y_1)*(heaviside_r_l)
        dhy_1 = y_dy_1 +(-y_dy_1)*(heaviside_r_u)+(-y_dy_1)*(heaviside_r_l)
        ddhy_1 = y_ddy_1 +(-y_ddy_1)*(heaviside_r_u)-(y_ddy_1)*(heaviside_r_l)


        # plt.figure()
        # plt.scatter(hx_1.detach().numpy(),hy_1.detach().numpy())
        # plt.scatter(hx_1[indices_1].detach().numpy(),hy_1[indices_1].detach().numpy(),c="red")
        # plt.scatter(hx_2.detach().numpy(),hy_2.detach().numpy())
        # # plt.scatter(hx_2[indices_2].detach().numpy(),hy_2[indices_2].detach().numpy(),c="red")
        # plt.show()
        


        support_function_matrix = np.array([[1,init_time],[1,final_time]])
        coefficients_matrix = torch.tensor(np.linalg.inv(support_function_matrix),dtype=torch.float)
        
        free_support_function_matrix = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train))
        d_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape)))
        
        phis = torch.matmul(free_support_function_matrix,coefficients_matrix)
        phi1 = phis[:,0].reshape(len(self.x_train),1)
        phi2 = phis[:,1].reshape(len(self.x_train),1)
        d_phis = torch.matmul(d_free_support_function_matrix,coefficients_matrix)
        d_phi1 = d_phis[:,0].reshape(len(self.x_train),1)
        d_phi2 = d_phis[:,1].reshape(len(self.x_train),1)
        
        
        phi1_theta1_init = phi1*init_theta_1
        phi1_theta2_init = phi1*init_theta_2
        phi2_theta1_final = phi2*final_theta_1
        phi2_theta2_final = phi2*final_theta_2
       
        d_phi1_theta1_init = d_phi1*init_theta_1
        d_phi1_theta2_init = d_phi1*init_theta_2
        d_phi2_theta1_final = d_phi2*final_theta_1
        d_phi2_theta2_final = d_phi2*final_theta_2


        htheta_1 = torch.matmul(h.add(phi1_h1).add(phi2_hf),btheta_1).reshape(self.x_train.shape).add(phi1_theta1_init).add(phi2_theta1_final)
        dhtheta_1 = self.c*torch.matmul(dh.add(d_phi1_h1).add(d_phi2_hf),btheta_1).reshape(self.x_train.shape).add(d_phi1_theta1_init).add(d_phi2_theta1_final)

        hdelta_1 = torch.matmul(h,bdelta_1).reshape(self.x_train.shape)
        dhdelta_1 = torch.matmul(dh,bdelta_1).reshape(self.x_train.shape)

        htheta_2 = torch.matmul(h.add(phi1_h1).add(phi2_hf),btheta_2).reshape(self.x_train.shape).add(phi1_theta2_init).add(phi2_theta2_final)
        dhtheta_2 = self.c*torch.matmul(dh.add(d_phi1_h1).add(d_phi2_hf),btheta_2).reshape(self.x_train.shape).add(d_phi1_theta2_init).add(d_phi2_theta2_final)

        hdelta_2 = torch.matmul(h,bdelta_2).reshape(self.x_train.shape)
        dhdelta_2 = torch.matmul(dh,bdelta_2).reshape(self.x_train.shape)

        
        lambda_dx_1 = torch.matmul(h,blambda_x_1).reshape(self.x_train.shape)
        d_lambda_dx_1 = torch.matmul(dh,blambda_x_1).reshape(self.x_train.shape)
        lambda_dy_1 = torch.matmul(h,blambda_y_1).reshape(self.x_train.shape)
        d_lambda_dy_1 = torch.matmul(dh,blambda_y_1).reshape(self.x_train.shape)
        lambda_ddx_1 = torch.matmul(h,blambda_dx_1).reshape(self.x_train.shape)
        d_lambda_ddx_1 = torch.matmul(dh,blambda_dx_1).reshape(self.x_train.shape)
        lambda_ddy_1= torch.matmul(h,blambda_dy_1).reshape(self.x_train.shape)
        d_lambda_ddy_1= torch.matmul(dh,blambda_dy_1).reshape(self.x_train.shape)
        lambda_dtheta_1 = torch.matmul(h,blambda_dtheta_1).reshape(self.x_train.shape)
        d_lambda_dtheta_1 = torch.matmul(dh,blambda_dtheta_1).reshape(self.x_train.shape)
        lambda_ddelta_1 = torch.matmul(h,blambda_ddelta_1).reshape(self.x_train.shape)
        d_lambda_ddelta_1 = torch.matmul(dh,blambda_ddelta_1).reshape(self.x_train.shape)


        lambda_dx_2 = torch.matmul(h,blambda_x_2).reshape(self.x_train.shape)
        d_lambda_dx_2 = torch.matmul(dh,blambda_x_2).reshape(self.x_train.shape)
        lambda_dy_2 = torch.matmul(h,blambda_y_2).reshape(self.x_train.shape)
        d_lambda_dy_2 = torch.matmul(dh,blambda_y_2).reshape(self.x_train.shape)
        lambda_ddx_2 = torch.matmul(h,blambda_dx_2).reshape(self.x_train.shape)
        d_lambda_ddx_2 = torch.matmul(dh,blambda_dx_2).reshape(self.x_train.shape)
        lambda_ddy_2= torch.matmul(h,blambda_dy_2).reshape(self.x_train.shape)
        d_lambda_ddy_2= torch.matmul(dh,blambda_dy_2).reshape(self.x_train.shape)
        lambda_dtheta_2 = torch.matmul(h,blambda_dtheta_2).reshape(self.x_train.shape)
        d_lambda_dtheta_2 = torch.matmul(dh,blambda_dtheta_2).reshape(self.x_train.shape)
        lambda_ddelta_2 = torch.matmul(h,blambda_ddelta_2).reshape(self.x_train.shape)
        d_lambda_ddelta_2 = torch.matmul(dh,blambda_ddelta_2).reshape(self.x_train.shape)


        #define pre-computations to make your life happier

        v_1 = (dhx_1**2+dhy_1**2)**(1/2)
        dhv_x_1 = (v_1)**(-1/2)*dhx_1
        dhv_y_1 = (v_1)**(-1/2)*dhy_1
        dcos_slip_1_x = torch.sin(torch.atan(dhy_1/dhx_1))*(dhy_1*dhx_1**-2)/(1+(dhy_1/dhx_1)**2)
        dcos_slip_1_y = -torch.sin(torch.atan(dhy_1/dhx_1))*(dhx_1**-1)/(1+(dhy_1/dhx_1)**2)
        cos_theta_1 = torch.cos(htheta_1)
        sin_theta_1 = torch.sin(htheta_1)
        tan_delta_1 = torch.tan(hdelta_1)
        cos_slip_1 = torch.cos(torch.atan(dhy_1/dhx_1))

        v_2 = (dhx_2**2+dhy_2**2)**(1/2)
        dhv_x_2 = (v_2)**(-1/2)*dhx_2
        dhv_y_2 = (v_2)**(-1/2)*dhy_2
        dcos_slip_2_x = torch.sin(torch.atan(dhy_2/dhx_2))*(dhy_2*dhx_2**-2)/(1+(dhy_2/dhx_2)**2)
        dcos_slip_2_y = -torch.sin(torch.atan(dhy_2/dhx_2))*(dhx_2**-1)/(1+(dhy_2/dhx_2)**2)
      
        
        cos_theta_2 = torch.cos(htheta_2)
        sin_theta_2 = torch.sin(htheta_2)
        tan_delta_2 = torch.tan(hdelta_2)
        cos_slip_2 = torch.cos(torch.atan(dhy_2/dhx_2))

        l_dx_1 = dhx_1 - v_1*cos_theta_1
        l_dy_1 = dhy_1 - v_1*sin_theta_1
        l_ddx_1 = ddhx_1+lambda_ddx_1
        l_ddy_1 = ddhy_1+lambda_ddy_1
        l_dtheta_1 = dhtheta_1 - (v_1*tan_delta_1*cos_slip_1/self.l)
        l_ddelta_1 = dhdelta_1+lambda_ddelta_1
        
        # l_lambda_dx_1 = d_lambda_dx_1 -(4*((hx_1-hx_2)**2-self.d**2)*(hx_1-hx_2))
        # l_lambda_dy_1 = d_lambda_dy_1 -(4*((hy_1-hy_2)**2-self.d**2)*(hy_1-hy_2))
        l_lambda_dx_1 = d_lambda_dx_1
        l_lambda_dy_1 = d_lambda_dy_1
        l_lambda_ddx_1 = d_lambda_ddx_1 -(-lambda_dx_1*cos_theta_1*dhv_x_1-lambda_dy_1*sin_theta_1*dhv_x_1-lambda_dtheta_1*tan_delta_1/self.l*(dhv_x_1*cos_slip_1+dcos_slip_1_x*v_1))
        l_lambda_ddy_1 = d_lambda_ddy_1 -(-lambda_dx_1*cos_theta_1*dhv_y_1-lambda_dy_1*sin_theta_1*dhv_y_1-lambda_dtheta_1*tan_delta_1/self.l*(dhv_y_1*cos_slip_1+dcos_slip_1_y*v_1))
        l_lambda_dtheta_1 = d_lambda_dtheta_1 - ( lambda_dx_1*v_1*sin_theta_1-lambda_dy_1*v_1*cos_theta_1)
        l_lambda_ddelta_1 = d_lambda_ddelta_1 - ( -lambda_dtheta_1/self.l*cos_slip_1*v_1*(1/torch.cos(hdelta_1))**2)


        l_dx_2 = dhx_2 - v_2*cos_theta_1
        l_dy_2 = dhy_2 - v_2*sin_theta_1
        l_ddx_2 = ddhx_2+lambda_ddx_2
        l_ddy_2 = ddhy_2+lambda_ddy_2
        l_dtheta_2 = dhtheta_2 - (v_2*tan_delta_2*cos_slip_2/self.l)
        l_ddelta_2 = dhdelta_2+lambda_ddelta_2
        
        
        # l_lambda_dx_2 = d_lambda_dx_2 +(4*((hx_1-hx_2)**2-self.d**2)*(hx_1-hx_2))
        # l_lambda_dy_2 = d_lambda_dy_2 +(4*((hy_1-hy_2)**2-self.d**2)*(hy_1-hy_2))
        l_lambda_dx_2 = d_lambda_dx_2
        l_lambda_dy_2 = d_lambda_dy_2
        l_lambda_ddx_2 = d_lambda_ddx_2 -(-lambda_dx_2*cos_theta_2*dhv_x_2-lambda_dy_2*sin_theta_2*dhv_x_2-lambda_dtheta_2*tan_delta_2/self.l*(dhv_x_2*cos_slip_2+dcos_slip_2_x*v_2))
        l_lambda_ddy_2 = d_lambda_ddy_2 -(-lambda_dx_2*cos_theta_2*dhv_y_2-lambda_dy_2*sin_theta_2*dhv_y_2-lambda_dtheta_2*tan_delta_2/self.l*(dhv_y_2*cos_slip_2+dcos_slip_2_y*v_2))
        l_lambda_dtheta_2 = d_lambda_dtheta_2 -(lambda_dx_2*v_2*sin_theta_2-lambda_dy_2*v_2*cos_theta_2)
        l_lambda_ddelta_2 = d_lambda_ddelta_2 -(-lambda_dtheta_2/self.l*cos_slip_2*v_2*(1/torch.cos(hdelta_2))**2)
        
        
        loss= torch.vstack((  
                              l_dx_1,
                              l_dy_1,
                              l_ddx_1,
                              l_ddy_1,
                              l_dtheta_1,
                              l_ddelta_1,
                              l_lambda_dx_1,
                              l_lambda_dy_1,
                              l_lambda_ddx_1,
                              l_lambda_ddy_1,
                              l_lambda_dtheta_1,
                              l_lambda_ddelta_1,
                              l_dx_2,
                              l_dy_2,
                              l_ddx_2,
                              l_ddy_2,
                              l_dtheta_2,
                              l_ddelta_2,
                              l_lambda_dx_2,
                              l_lambda_dy_2,
                              l_lambda_ddx_2,
                              l_lambda_ddy_2,
                              l_lambda_dtheta_2,
                              l_lambda_ddelta_2
                              ))  

        return loss
            
    def predict_loss(self,betas):

        betas = self.betas        
        h = self.get_h(self.x_train)
        dh = self.get_dh(self.x_train)
        dhh = self.get_dhh(self.x_train)

        bx_1 = betas[0:self.nodes]
        by_1 = betas[self.nodes:self.nodes*2]
        btheta_1 = betas[self.nodes*2:self.nodes*3]
        bdelta_1 = betas[self.nodes*3:self.nodes*4]
        blambda_x_1 = betas[self.nodes*4:self.nodes*5]
        blambda_y_1 = betas[self.nodes*5:self.nodes*6]
        blambda_dx_1 = betas[self.nodes*6:self.nodes*7]
        blambda_dy_1 = betas[self.nodes*7:self.nodes*8]
        blambda_dtheta_1 = betas[self.nodes*8:self.nodes*9]
        blambda_ddelta_1 = betas[self.nodes*9:self.nodes*10]
        
        bx_2 = betas[self.nodes*10:self.nodes*11]
        by_2 = betas[self.nodes*11:self.nodes*12]
        btheta_2 = betas[self.nodes*12:self.nodes*13]
        bdelta_2 = betas[self.nodes*13:self.nodes*14]
        blambda_x_2 = betas[self.nodes*14:self.nodes*15]
        blambda_y_2 = betas[self.nodes*15:self.nodes*16]
        blambda_dx_2 = betas[self.nodes*16:self.nodes*17]
        blambda_dy_2 = betas[self.nodes*17:self.nodes*18]
        blambda_dtheta_2 = betas[self.nodes*18:self.nodes*19]
        blambda_ddelta_2 = betas[self.nodes*19:self.nodes*20]

        init_x_1 = self.y_train_1[0,0]        
        final_x_1 = self.y_train_1[-1,0]
        init_y_1 = self.y_train_1[0,1]
        final_y_1 = self.y_train_1[-1,1]
        init_theta_1 = self.y_train_1[0,2]        
        final_theta_1 = self.y_train_1[-1,2]
        init_vx_1 = self.y_train_1[0,3]        
        init_vy_1 = self.y_train_1[0,4]
        init_ax_1 = self.y_train_1[0,5]        
        init_ay_1 = self.y_train_1[0,6]
        init_h_1 = self.get_h(self.x_train[0])
        final_h_1 = self.get_h(self.x_train[-1])
        init_dh_1 = self.get_dh(self.x_train[0])
        init_dhh_1 = self.get_dhh(self.x_train[0])
        

        init_x_2 = self.y_train_2[0,0]        
        final_x_2 = self.y_train_2[-1,0]
        init_y_2 = self.y_train_2[0,1]
        final_y_2 = self.y_train_2[-1,1]
        init_theta_2 = self.y_train_2[0,2]        
        final_theta_2 = self.y_train_2[-1,2]
        init_vx_2 = self.y_train_2[0,3]        
        init_vy_2 = self.y_train_2[0,4]
        init_ax_2 = self.y_train_2[0,5]        
        init_ay_2 = self.y_train_2[0,6]

        cross_x_2 = self.y_train_2[1,0]
        cross_y_2 = self.y_train_2[1,1]
        r1_x_2 =self.y_train_2[2,0]
        r1_y_2 =self.y_train_2[2,1]
        r2_x_2 =self.y_train_2[3,0]
        r2_y_2 =self.y_train_2[3,1]
        r3_x_2 =self.y_train_2[4,0]
        r3_y_2 =self.y_train_2[4,1]

        rel_cross =  self.y_train_2[1,-1]
        rel_r1 = self.y_train_2[2,-1]
        rel_r2 = self.y_train_2[3,-1]
        rel_r3 = self.y_train_2[4,-1]

        final_time = self.x_train[-1].numpy()[0]
        init_time = self.x_train[0].numpy()[0]
        
        
        cross_time= ((final_time-init_time)*rel_cross+init_time).reshape(self.x_train[0].shape)
        r1_time= ((final_time-init_time)*rel_r1+init_time).reshape(self.x_train[0].shape)
        r2_time= ((final_time-init_time)*rel_r2+init_time).reshape(self.x_train[0].shape)
        r3_time= ((final_time-init_time)*rel_r3+init_time).reshape(self.x_train[0].shape)
        
        cross_index = np.where(self.x_train>=cross_time.numpy()[0])[0][0]
        cross_time = self.x_train[cross_index]
        r1_index = np.where(self.x_train>=r1_time.numpy()[0])[0][0]
        r1_time = self.x_train[r1_index]
        r2_index = np.where(self.x_train>=r2_time.numpy()[0])[0][0]
        r2_time = self.x_train[r2_index]
        r3_index = np.where(self.x_train>=r3_time.numpy()[0])[0][0]
        r3_time = self.x_train[r3_index]
        

        init_h_2 = self.get_h(self.x_train[0])
        cross_h_2 = self.get_h(cross_time)
        r1_h_2 = self.get_h(r1_time)
        r2_h_2 = self.get_h(r2_time)
        r3_h_2 = self.get_h(r3_time)
        final_h_2 = self.get_h(self.x_train[-1])
        init_dh_2 = self.get_dh(self.x_train[0])
        init_ddh_2 = self.get_dhh(self.x_train[0])
        cross_time = cross_time.numpy()[0]
        r1_time = r1_time.numpy()[0]
        r2_time = r2_time.numpy()[0]
        r3_time = r3_time.numpy()[0]
        
        
        # x_linspace[0:len(self.x_train)/2]=torch.linspace(init_x_2,final_x_2,int(abs(final_x_2.detach().numpy()-init_x_2.detach().numpy())/2))
        # x_linspace[len(self.x_train)/2:]= final_x_2
        # y_linspace[len(self.x_train)/2:]=torch.linspace(init_y_2,final_y_2,int(abs(final_y_2.detach().numpy()-init_y_2.detach().numpy())/2))
        # y_linspace[0:len(self.x_train)/2]= init_y_2
        # plt.figure()
        # plt.plot(x_linspace.detach().numpy(),plt.plot(y_linspace.detach().numpy()))
        # plt.show()
        """""
        For a 4 point constrained problem on x and y  we have: 

        Consider support functions the polynomials: t^0,t^1,t^2,t^3: Then for x and y we have:
        x(0)       [[1,0,0,0]
        x(f)       [1,t,t^2,t^3]
        xdot(0)    [0,1,0,0]
        xdotdot(0) [0,0,2,0]]

        Consider support functions the polynomials: t^0,t^1,t^2,t^3,t^4,t^5: Then for x and y we have:
        x(0)       [[1,t(0),t(0)^2,t(0)^3,t(0)^4,t(0)^5]
        x(r1)      [1,t(r1),t(r1)^2,t(r1)^3,t(r1)^4,t(r1)^5]
        x(r3)      [1,t(r3),t(r3)^2,t(r3)^3,t(r3)^4,t(r3)^5]
        x(f)       [1,t(rf),t(rf)^2,t(rf)^3]
        xdot(0)    [0,1,0,0]
        xdotdot(0) [0,0,2,0]]
        

        For a 2 point constrained problem on theta we have:
        Consider support functions the polynomials: t^0,t^1: Then for theta have:
        theta(0)       [[1,0]
        theta(f)       [1,t]
        """""
        
        support_function_matrix = np.array([[1,init_time,init_time**2,init_time**3],[1,final_time,final_time**2,final_time**3],[0,1,2*init_time,3*init_time**2],[0,0,2,6*init_time]])
        coefficients_matrix = torch.tensor(np.linalg.inv(support_function_matrix),dtype=torch.float)
        
        free_support_function_matrix = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train,self.x_train**2,self.x_train**3))
        d_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape),2*self.x_train,3*self.x_train**2))
        dd_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.zeros(size=self.x_train.shape),2*torch.ones(size=self.x_train.shape),6*self.x_train))

        
        support_function_matrix_2 = np.array([[1,init_time,init_time**2,init_time**3,init_time**4,init_time**5,init_time**6,init_time**7]\
                                              ,[1,final_time,final_time**2,final_time**3,final_time**4,final_time**5,final_time**6,final_time**7]\
                                              ,[1,cross_time,cross_time**2,cross_time**3,cross_time**4,cross_time**5,cross_time**6,cross_time**7]\
                                              ,[1,r1_time,r1_time**2,r1_time**3,r1_time**4,r1_time**5,r1_time**6,r1_time**7]\
                                              ,[1,r2_time,r2_time**2,r2_time**3,r2_time**4,r2_time**5,r2_time**6,r2_time**7]\
                                              ,[1,r3_time,r3_time**2,r3_time**3,r3_time**4,r3_time**5,r3_time**6,r3_time**7]\
                                              ,[0,1,2*init_time,3*init_time**2,4*init_time**3,5*init_time**4,6*init_time**5,7*init_time**6]\
                                              ,[0,0,2,6*init_time,12*init_time**2,20*init_time**3,30*init_time**4,42*init_time**5]])
        
        coefficients_matrix_2 = torch.tensor(np.linalg.inv(support_function_matrix_2),dtype=torch.float)
        free_support_function_matrix_2 = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train,self.x_train**2,self.x_train**3,self.x_train**4,self.x_train**5,self.x_train**6,self.x_train**7))
        d_free_support_function_matrix_2 = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape),2*self.x_train,3*self.x_train**2\
                                                         ,4*self.x_train**3,5*self.x_train**4,6*self.x_train**5,7*self.x_train**6))
        dd_free_support_function_matrix_2 = torch.hstack((torch.zeros(size=self.x_train.shape),torch.zeros(size=self.x_train.shape),2*torch.ones(size=self.x_train.shape)\
                                                          ,6*self.x_train,12*self.x_train**2,20*self.x_train**3,30*self.x_train**4,42*self.x_train**5))

        
        phis = torch.matmul(free_support_function_matrix,coefficients_matrix)
        phi1 = phis[:,0].reshape(len(self.x_train),1)
        phi2 = phis[:,1].reshape(len(self.x_train),1)
        phi3 = phis[:,2].reshape(len(self.x_train),1)
        phi4 = phis[:,3].reshape(len(self.x_train),1)
        d_phis = torch.matmul(d_free_support_function_matrix,coefficients_matrix)
        d_phi1 = d_phis[:,0].reshape(len(self.x_train),1)
        d_phi2 = d_phis[:,1].reshape(len(self.x_train),1)
        d_phi3 = d_phis[:,2].reshape(len(self.x_train),1)
        d_phi4 = d_phis[:,3].reshape(len(self.x_train),1)
        dd_phis = torch.matmul(dd_free_support_function_matrix,coefficients_matrix)
        dd_phi1 = dd_phis[:,0].reshape(len(self.x_train),1)
        dd_phi2 = dd_phis[:,1].reshape(len(self.x_train),1)
        dd_phi3 = dd_phis[:,2].reshape(len(self.x_train),1)
        dd_phi4 = dd_phis[:,3].reshape(len(self.x_train),1)

        
        phi1_h1 = torch.matmul(-phi1,init_h_1)
        phi1_x1_init = phi1*init_x_1
        phi1_y1_init = phi1*init_y_1
        phi2_hf = torch.matmul(-phi2,final_h_1)
        phi2_x1_final = phi2*final_x_1
        phi2_y1_final = phi2*final_y_1
        phi3_dh1 = torch.matmul(-phi3,init_dh_1)
        phi3_vx1_init = phi3*init_vx_1
        phi3_vy1_init = phi3*init_vy_1
        phi4_ddh1 = torch.matmul(-phi4,init_dhh_1)
        phi4_ax1_init = phi4*init_ax_1
        phi4_ay1_init = phi4*init_ay_1
        
        d_phi1_h1 = torch.matmul(-d_phi1,init_h_1)
        d_phi1_x1_init = d_phi1*init_x_1
        d_phi1_y1_init = d_phi1*init_y_1
        d_phi2_hf = torch.matmul(-d_phi2,final_h_1)
        d_phi2_x1_final = d_phi2*final_x_1
        d_phi2_y1_final = d_phi2*final_y_1
        d_phi3_dh1 = torch.matmul(-d_phi3,init_dh_1)
        d_phi3_vx1_init = d_phi3*init_vx_1
        d_phi3_vy1_init = d_phi3*init_vy_1
        d_phi4_ddh1 = torch.matmul(-d_phi4,init_dhh_1)
        d_phi4_ax1_init = d_phi4*init_ax_1
        d_phi4_ay1_init = d_phi4*init_ay_1
        
        dd_phi1_h1 = torch.matmul(-dd_phi1,init_h_1)
        dd_phi1_x1_init = dd_phi1*init_x_1
        dd_phi1_y1_init = dd_phi1*init_y_1
        dd_phi2_hf = torch.matmul(-dd_phi2,final_h_1)
        dd_phi2_x1_final = dd_phi2*final_x_1
        dd_phi2_y1_final = dd_phi2*final_y_1
        dd_phi3_dh1 = torch.matmul(-dd_phi3,init_dh_1)
        dd_phi3_vx1_init = dd_phi3*init_vx_1
        dd_phi3_vy1_init = dd_phi3*init_vy_1
        dd_phi4_ddh1 = torch.matmul(-dd_phi4,init_dhh_1)
        dd_phi4_ax1_init = dd_phi4*init_ax_1
        dd_phi4_ay1_init = dd_phi4*init_ay_1
        
   

        phis_2 = torch.matmul(free_support_function_matrix_2,coefficients_matrix_2)
        phi1_2 = phis_2[:,0].reshape(len(self.x_train),1)
        phi2_2 = phis_2[:,1].reshape(len(self.x_train),1)
        phi3_2 = phis_2[:,2].reshape(len(self.x_train),1)
        phi4_2 = phis_2[:,3].reshape(len(self.x_train),1)
        phi5_2 = phis_2[:,4].reshape(len(self.x_train),1)
        phi6_2 = phis_2[:,5].reshape(len(self.x_train),1)
        phi7_2 = phis_2[:,6].reshape(len(self.x_train),1)
        phi8_2 = phis_2[:,7].reshape(len(self.x_train),1)

        d_phis_2 = torch.matmul(d_free_support_function_matrix_2,coefficients_matrix_2)
        d_phi1_2 = d_phis_2[:,0].reshape(len(self.x_train),1)
        d_phi2_2 = d_phis_2[:,1].reshape(len(self.x_train),1)
        d_phi3_2 = d_phis_2[:,2].reshape(len(self.x_train),1)
        d_phi4_2 = d_phis_2[:,3].reshape(len(self.x_train),1)
        d_phi5_2 = d_phis_2[:,4].reshape(len(self.x_train),1)
        d_phi6_2 = d_phis_2[:,5].reshape(len(self.x_train),1)
        d_phi7_2 = d_phis_2[:,6].reshape(len(self.x_train),1)
        d_phi8_2 = d_phis_2[:,7].reshape(len(self.x_train),1)
        
        dd_phis_2 = torch.matmul(dd_free_support_function_matrix_2,coefficients_matrix_2)
        dd_phi1_2 = dd_phis_2[:,0].reshape(len(self.x_train),1)
        dd_phi2_2 = dd_phis_2[:,1].reshape(len(self.x_train),1)
        dd_phi3_2 = dd_phis_2[:,2].reshape(len(self.x_train),1)
        dd_phi4_2 = dd_phis_2[:,3].reshape(len(self.x_train),1)
        dd_phi5_2 = dd_phis_2[:,4].reshape(len(self.x_train),1)
        dd_phi6_2 = dd_phis_2[:,5].reshape(len(self.x_train),1)
        dd_phi7_2 = dd_phis_2[:,6].reshape(len(self.x_train),1)
        dd_phi8_2 = dd_phis_2[:,7].reshape(len(self.x_train),1)

        print(phi1_2.shape)
        print(init_h_2.shape)
        phi1_h1_2 = torch.matmul(-phi1_2,init_h_2)
        phi1_x2_init_2 = phi1_2*init_x_2
        phi1_y2_init_2 = phi1_2*init_y_2
        phi2_hf_2 = torch.matmul(-phi2_2,final_h_2)
        phi2_x2_final_2 = phi2_2*final_x_2
        phi2_y2_final_2 = phi2_2*final_y_2
        phi3_hcross_2 = torch.matmul(-phi3_2,cross_h_2)
        phi3_x2_cross_2 = phi3_2*cross_x_2
        phi3_y2_cross_2 = phi3_2*cross_y_2
        phi4_hr1_2 = torch.matmul(-phi4_2,r1_h_2)
        phi4_x2_r1_2 = phi4_2*r1_x_2
        phi4_y2_r1_2 = phi4_2*r1_y_2
        phi5_hr2_2 = torch.matmul(-phi5_2,r2_h_2)
        phi5_x2_r2_2 = phi5_2*r2_x_2
        phi5_y2_r2_2 = phi5_2*r2_y_2
        phi6_hr3_2 = torch.matmul(-phi6_2,r3_h_2)
        phi6_x2_r3_2 = phi6_2*r3_x_2
        phi6_y2_r3_2 = phi6_2*r3_y_2
        phi7_dh2_2 = torch.matmul(-phi7_2,init_dh_2)
        phi7_vx2_init_2 = phi7_2*init_vx_2
        phi7_vy2_init_2 = phi7_2*init_vy_2
        phi8_ddh2_2 = torch.matmul(-phi8_2,init_ddh_2)
        phi8_ax2_init_2 = phi8_2*init_ax_2
        phi8_ay2_init_2 = phi8_2*init_ay_2
        
        d_phi1_h1_2 = torch.matmul(-d_phi1_2,init_h_2)
        d_phi1_x2_init_2 = d_phi1_2*init_x_2
        d_phi1_y2_init_2 = d_phi1_2*init_y_2
        d_phi2_hf_2 = torch.matmul(-d_phi2_2,final_h_2)
        d_phi2_x2_final_2 = d_phi2_2*final_x_2
        d_phi2_y2_final_2 = d_phi2_2*final_y_2
        d_phi3_hcross_2 = torch.matmul(-d_phi3_2,cross_h_2)
        d_phi3_x2_cross_2 = d_phi3_2*cross_x_2
        d_phi3_y2_cross_2 = d_phi3_2*cross_y_2
        d_phi4_hr1_2 = torch.matmul(-d_phi4_2,r1_h_2)
        d_phi4_x2_r1_2 = d_phi4_2*r1_x_2
        d_phi4_y2_r1_2 = d_phi4_2*r1_y_2
        d_phi5_hr2_2 = torch.matmul(-d_phi5_2,r2_h_2)
        d_phi5_x2_r2_2 = d_phi5_2*r2_x_2
        d_phi5_y2_r2_2 = d_phi5_2*r2_y_2
        d_phi6_hr3_2 = torch.matmul(-d_phi6_2,r3_h_2)
        d_phi6_x2_r3_2 = d_phi6_2*r3_x_2
        d_phi6_y2_r3_2 = d_phi6_2*r3_y_2
        d_phi7_dh2_2 = torch.matmul(-d_phi7_2,init_dh_2)
        d_phi7_vx2_init_2 = d_phi7_2*init_vx_2
        d_phi7_vy2_init_2 = d_phi7_2*init_vy_2
        d_phi8_ddh2_2 = torch.matmul(-d_phi8_2,init_ddh_2)
        d_phi8_ax2_init_2 = d_phi8_2*init_ax_2
        d_phi8_ay2_init_2 = d_phi8_2*init_ay_2

        dd_phi1_h1_2 = torch.matmul(-dd_phi1_2,init_h_2)
        dd_phi1_x2_init_2 = dd_phi1_2*init_x_2
        dd_phi1_y2_init_2 = dd_phi1_2*init_y_2
        dd_phi2_hf_2 = torch.matmul(-dd_phi2_2,final_h_2)
        dd_phi2_x2_final_2 = dd_phi2_2*final_x_2
        dd_phi2_y2_final_2 = dd_phi2_2*final_y_2
        dd_phi3_hcross_2 = torch.matmul(-dd_phi3_2,cross_h_2)
        dd_phi3_x2_cross_2 = dd_phi3_2*cross_x_2
        dd_phi3_y2_cross_2 = dd_phi3_2*cross_y_2
        dd_phi4_hr1_2 = torch.matmul(-dd_phi4_2,r1_h_2)
        dd_phi4_x2_r1_2 = dd_phi4_2*r1_x_2
        dd_phi4_y2_r1_2 = dd_phi4_2*r1_y_2
        dd_phi5_hr2_2 = torch.matmul(-dd_phi5_2,r2_h_2)
        dd_phi5_x2_r2_2 = dd_phi5_2*r2_x_2
        dd_phi5_y2_r2_2 = dd_phi5_2*r2_y_2
        dd_phi6_hr3_2 = torch.matmul(-dd_phi6_2,r3_h_2)
        dd_phi6_x2_r3_2 = dd_phi6_2*r3_x_2
        dd_phi6_y2_r3_2 = dd_phi6_2*r3_y_2
        dd_phi7_dh2_2 = torch.matmul(-dd_phi7_2,init_dh_2)
        dd_phi7_vx2_init_2 = dd_phi7_2*init_vx_2
        dd_phi7_vy2_init_2 = dd_phi7_2*init_vy_2
        dd_phi8_ddh2_2 = torch.matmul(-dd_phi8_2,init_ddh_2)
        dd_phi8_ax2_init_2 = dd_phi8_2*init_ax_2
        dd_phi8_ay2_init_2 = dd_phi8_2*init_ay_2



        
        hx_1 = torch.matmul(h.add(phi1_h1).add(phi2_hf).add(phi3_dh1).add(phi4_ddh1),bx_1).reshape(self.x_train.shape)\
            .add(phi1_x1_init).add(phi2_x1_final).add(phi3_vx1_init/self.c).add(phi4_ax1_init/self.c**2)
        
        dhx_1 = self.c*torch.matmul(dh.add(d_phi1_h1).add(d_phi2_hf).add(d_phi3_dh1).add(d_phi4_ddh1),bx_1).reshape(self.x_train.shape)\
            .add(d_phi1_x1_init).add(d_phi2_x1_final).add(d_phi3_vx1_init/self.c).add(d_phi4_ax1_init/self.c**2)
        
        ddhx_1 = self.c**2*torch.matmul(dhh.add(dd_phi1_h1).add(dd_phi2_hf).add(dd_phi3_dh1).add(dd_phi4_ddh1),bx_1).reshape(self.x_train.shape)\
            .add(dd_phi1_x1_init).add(dd_phi2_x1_final).add(dd_phi3_vx1_init/self.c).add(dd_phi4_ax1_init/self.c**2)
        
        hy_1 = torch.matmul(h.add(phi1_h1).add(phi2_hf).add(phi3_dh1).add(phi4_ddh1),by_1).reshape(self.x_train.shape)\
            .add(phi1_y1_init).add(phi2_y1_final).add(phi3_vy1_init/self.c).add(phi4_ay1_init/self.c**2)
        
        dhy_1 = self.c*torch.matmul(dh.add(d_phi1_h1).add(d_phi2_hf).add(d_phi3_dh1).add(d_phi4_ddh1),by_1).reshape(self.x_train.shape)\
            .add(d_phi1_y1_init).add(d_phi2_y1_final).add(d_phi3_vy1_init/self.c).add(d_phi4_ay1_init/self.c**2)
        
        ddhy_1 = self.c**2*torch.matmul(dhh.add(dd_phi1_h1).add(dd_phi2_hf).add(dd_phi3_dh1).add(dd_phi4_ddh1),by_1).reshape(self.x_train.shape)\
            .add(dd_phi1_y1_init).add(dd_phi2_y1_final).add(dd_phi3_vy1_init/self.c).add(dd_phi4_ay1_init/self.c**2)
        
        hx_2 = torch.matmul(h.add(phi1_h1_2).add(phi2_hf_2).add(phi3_hcross_2).add(phi4_hr1_2).add(phi5_hr2_2).add(phi6_hr3_2).add(phi7_dh2_2).add(phi8_ddh2_2),bx_2).reshape(self.x_train.shape)\
            .add(phi1_x2_init_2).add(phi2_x2_final_2).add(phi3_x2_cross_2).add(phi4_x2_r1_2).add(phi5_x2_r2_2).add(phi6_x2_r3_2).add(phi7_vx2_init_2/self.c).add(phi8_ax2_init_2/self.c**2)
        
        dhx_2 = self.c*torch.matmul(dh.add(d_phi1_h1_2).add(d_phi2_hf_2).add(d_phi3_hcross_2).add(d_phi4_hr1_2).add(d_phi5_hr2_2).add(d_phi6_hr3_2).add(d_phi7_dh2_2).add(d_phi8_ddh2_2),bx_2).reshape(self.x_train.shape)\
            .add(d_phi1_x2_init_2).add(d_phi2_x2_final_2).add(d_phi3_x2_cross_2).add(d_phi4_x2_r1_2).add(d_phi5_x2_r2_2).add(d_phi6_x2_r3_2).add(d_phi7_vx2_init_2/self.c).add(d_phi8_ax2_init_2/self.c**2)

        ddhx_2 = self.c**2*torch.matmul(dhh.add(dd_phi1_h1_2).add(dd_phi2_hf_2).add(dd_phi3_hcross_2).add(dd_phi4_hr1_2).add(dd_phi5_hr2_2).add(dd_phi6_hr3_2).add(dd_phi7_dh2_2).add(dd_phi8_ddh2_2),bx_2).reshape(self.x_train.shape)\
            .add(dd_phi1_x2_init_2).add(dd_phi2_x2_final_2).add(dd_phi3_x2_cross_2).add(dd_phi4_x2_r1_2).add(dd_phi5_x2_r2_2).add(dd_phi6_x2_r3_2).add(dd_phi7_vx2_init_2/self.c).add(dd_phi8_ax2_init_2/self.c**2)

        hy_2 = torch.matmul(h.add(phi1_h1_2).add(phi2_hf_2).add(phi3_hcross_2).add(phi4_hr1_2).add(phi5_hr2_2).add(phi6_hr3_2).add(phi7_dh2_2).add(phi8_ddh2_2),by_2).reshape(self.x_train.shape)\
            .add(phi1_y2_init_2).add(phi2_y2_final_2).add(phi3_y2_cross_2).add(phi4_y2_r1_2).add(phi5_y2_r2_2).add(phi6_y2_r3_2).add(phi7_vy2_init_2/self.c).add(phi8_ay2_init_2/self.c**2)
        
        dhy_2 = self.c*torch.matmul(dh.add(d_phi1_h1_2).add(d_phi2_hf_2).add(d_phi3_hcross_2).add(d_phi4_hr1_2).add(d_phi5_hr2_2).add(d_phi6_hr3_2).add(d_phi7_dh2_2).add(d_phi8_ddh2_2),by_2).reshape(self.x_train.shape)\
            .add(d_phi1_y2_init_2).add(d_phi2_y2_final_2).add(d_phi3_y2_cross_2).add(d_phi4_y2_r1_2).add(d_phi5_y2_r2_2).add(d_phi6_y2_r3_2).add(d_phi7_vy2_init_2/self.c).add(d_phi8_ay2_init_2/self.c**2)

        ddhy_2 = self.c**2*torch.matmul(dhh.add(dd_phi1_h1_2).add(dd_phi2_hf_2).add(dd_phi3_hcross_2).add(dd_phi4_hr1_2).add(dd_phi5_hr2_2).add(dd_phi6_hr3_2).add(dd_phi7_dh2_2).add(dd_phi8_ddh2_2),by_2).reshape(self.x_train.shape)\
            .add(dd_phi1_y2_init_2).add(dd_phi2_y2_final_2).add(dd_phi3_y2_cross_2).add(dd_phi4_y2_r1_2).add(dd_phi5_y2_r2_2).add(dd_phi6_y2_r3_2).add(dd_phi7_vy2_init_2/self.c).add(dd_phi8_ay2_init_2/self.c**2)

        y_y_1 = hy_1
        y_dy_1 = dhy_1
        y_ddy_1 = ddhy_1
        y_x_1 = hx_1
        y_dx_1 = dhx_1
        y_ddx_1 = ddhx_1
        y_y_2 = hy_2
        y_dy_2 = dhy_2
        y_ddy_2 = ddhy_2
        y_x_2 = hx_2
        y_dx_2 = dhx_2
        y_ddx_2 = ddhx_2
        heavy = lambda x,k: 1/2 +1/2*torch.tanh(k*x)
        x_c = 68
        y_c = 590.5

        #Add inequalities for x and y for car 1

        f_r_u_1 = init_y_1+self.d
        f_r_l_1 = init_y_1-self.d
        heaviside_r_u = heavy(y_y_1-f_r_u_1,self.k)
        heaviside_r_l =  heavy(f_r_l_1-y_y_1,self.k)

       
        
        min_dist =  y_x_2-y_x_1+y_y_2-y_y_1
        heaviside_min_dist = heavy(self.d-min_dist,self.k)
        heaviside_r1 = heavy(x_c-y_x_1,self.k)
        heaviside_r2 = heavy(y_y_2-y_c,self.k)
        region1 = heaviside_min_dist*heaviside_r1
        region2 = heaviside_min_dist*heaviside_r2
        
    
        # hx_1 = y_x_1 + (-(self.d+4.9)+y_x_2+y_y_2-y_y_1-y_x_1)*(region1)
        # dhx_1 = y_dx_1 + (y_dx_2+y_dy_2-y_dy_1-y_dx_1)*(region1)
        # ddhx_1 = y_ddx_1 + (y_ddx_2+y_ddy_2-y_ddy_1-y_ddx_1)*(region1)

        # hy_1 = y_y_1 + (-(self.d+4.9)+y_x_2-y_x_1+y_y_2-y_y_1)*(region1)
        # dhy_1 = y_dy_1 + (y_dx_2-y_dx_1+y_dy_2-y_dy_1)*(region1)
        # ddhy_1 = y_ddy_1 + (y_ddx_2-y_ddx_1+y_ddy_2-y_ddy_1)*(region1)
        
       
        hx_2 = y_x_2 + ((self.d+4.9)+y_x_1-y_y_2+y_y_1-y_x_2)*(region2)
        dhx_2 = y_dx_2 + (y_dx_1-y_dy_2+y_dy_1-y_dx_2)*(region2)
        ddhx_2 = y_ddx_2 + (y_ddx_1-y_ddy_2+y_ddy_1-y_ddx_2)*(region2)
      
        hy_2 = y_y_2 + ((self.d+4.9)+y_x_1-y_x_2+y_y_1-y_y_2)*(region2)
        dhy_2 = y_dy_2 + (y_dx_1-y_dx_2+y_dy_1-y_dy_2)*(region2)
        ddhy_2 = y_ddy_2 + (y_dx_1-y_dx_2+y_dy_1-y_dy_2)*(region2)

        y_y_1 = hy_1
        y_dy_1 = dhy_1
        y_ddy_1 = ddhy_1


        hy_1 = y_y_1 +(f_r_u_1-y_y_1)*(heaviside_r_u)+(f_r_l_1-y_y_1)*(heaviside_r_l)
        dhy_1 = y_dy_1 +(-y_dy_1)*(heaviside_r_u)+(-y_dy_1)*(heaviside_r_l)
        ddhy_1 = y_ddy_1 +(-y_ddy_1)*(heaviside_r_u)-(y_ddy_1)*(heaviside_r_l)


        # plt.figure()
        # plt.scatter(hx_1.detach().numpy(),hy_1.detach().numpy())
        # plt.scatter(hx_1[indices_1].detach().numpy(),hy_1[indices_1].detach().numpy(),c="red")
        # plt.scatter(hx_2.detach().numpy(),hy_2.detach().numpy())
        # # plt.scatter(hx_2[indices_2].detach().numpy(),hy_2[indices_2].detach().numpy(),c="red")
        # plt.show()
        
        

        support_function_matrix = np.array([[1,init_time],[1,final_time]])
        coefficients_matrix = torch.tensor(np.linalg.inv(support_function_matrix),dtype=torch.float)
        
        free_support_function_matrix = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train))
        d_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape)))
        
        phis = torch.matmul(free_support_function_matrix,coefficients_matrix)
        phi1 = phis[:,0].reshape(len(self.x_train),1)
        phi2 = phis[:,1].reshape(len(self.x_train),1)
        d_phis = torch.matmul(d_free_support_function_matrix,coefficients_matrix)
        d_phi1 = d_phis[:,0].reshape(len(self.x_train),1)
        d_phi2 = d_phis[:,1].reshape(len(self.x_train),1)
        
        
        phi1_theta1_init = phi1*init_theta_1
        phi1_theta2_init = phi1*init_theta_2
        phi2_theta1_final = phi2*final_theta_1
        phi2_theta2_final = phi2*final_theta_2
       
        d_phi1_theta1_init = d_phi1*init_theta_1
        d_phi1_theta2_init = d_phi1*init_theta_2
        d_phi2_theta1_final = d_phi2*final_theta_1
        d_phi2_theta2_final = d_phi2*final_theta_2


        htheta_1 = torch.matmul(h.add(phi1_h1).add(phi2_hf),btheta_1).reshape(self.x_train.shape).add(phi1_theta1_init).add(phi2_theta1_final)
        dhtheta_1 = self.c*torch.matmul(dh.add(d_phi1_h1).add(d_phi2_hf),btheta_1).reshape(self.x_train.shape).add(d_phi1_theta1_init).add(d_phi2_theta1_final)

        hdelta_1 = torch.matmul(h,bdelta_1).reshape(self.x_train.shape)
        dhdelta_1 = torch.matmul(dh,bdelta_1).reshape(self.x_train.shape)

        htheta_2 = torch.matmul(h.add(phi1_h1).add(phi2_hf),btheta_2).reshape(self.x_train.shape).add(phi1_theta2_init).add(phi2_theta2_final)
        dhtheta_2 = self.c*torch.matmul(dh.add(d_phi1_h1).add(d_phi2_hf),btheta_2).reshape(self.x_train.shape).add(d_phi1_theta2_init).add(d_phi2_theta2_final)

        hdelta_2 = torch.matmul(h,bdelta_2).reshape(self.x_train.shape)
        dhdelta_2 = torch.matmul(dh,bdelta_2).reshape(self.x_train.shape)

        
        lambda_dx_1 = torch.matmul(h,blambda_x_1).reshape(self.x_train.shape)
        d_lambda_dx_1 = torch.matmul(dh,blambda_x_1).reshape(self.x_train.shape)
        lambda_dy_1 = torch.matmul(h,blambda_y_1).reshape(self.x_train.shape)
        d_lambda_dy_1 = torch.matmul(dh,blambda_y_1).reshape(self.x_train.shape)
        lambda_ddx_1 = torch.matmul(h,blambda_dx_1).reshape(self.x_train.shape)
        d_lambda_ddx_1 = torch.matmul(dh,blambda_dx_1).reshape(self.x_train.shape)
        lambda_ddy_1= torch.matmul(h,blambda_dy_1).reshape(self.x_train.shape)
        d_lambda_ddy_1= torch.matmul(dh,blambda_dy_1).reshape(self.x_train.shape)
        lambda_dtheta_1 = torch.matmul(h,blambda_dtheta_1).reshape(self.x_train.shape)
        d_lambda_dtheta_1 = torch.matmul(dh,blambda_dtheta_1).reshape(self.x_train.shape)
        lambda_ddelta_1 = torch.matmul(h,blambda_ddelta_1).reshape(self.x_train.shape)
        d_lambda_ddelta_1 = torch.matmul(dh,blambda_ddelta_1).reshape(self.x_train.shape)


        lambda_dx_2 = torch.matmul(h,blambda_x_2).reshape(self.x_train.shape)
        d_lambda_dx_2 = torch.matmul(dh,blambda_x_2).reshape(self.x_train.shape)
        lambda_dy_2 = torch.matmul(h,blambda_y_2).reshape(self.x_train.shape)
        d_lambda_dy_2 = torch.matmul(dh,blambda_y_2).reshape(self.x_train.shape)
        lambda_ddx_2 = torch.matmul(h,blambda_dx_2).reshape(self.x_train.shape)
        d_lambda_ddx_2 = torch.matmul(dh,blambda_dx_2).reshape(self.x_train.shape)
        lambda_ddy_2= torch.matmul(h,blambda_dy_2).reshape(self.x_train.shape)
        d_lambda_ddy_2= torch.matmul(dh,blambda_dy_2).reshape(self.x_train.shape)
        lambda_dtheta_2 = torch.matmul(h,blambda_dtheta_2).reshape(self.x_train.shape)
        d_lambda_dtheta_2 = torch.matmul(dh,blambda_dtheta_2).reshape(self.x_train.shape)
        lambda_ddelta_2 = torch.matmul(h,blambda_ddelta_2).reshape(self.x_train.shape)
        d_lambda_ddelta_2 = torch.matmul(dh,blambda_ddelta_2).reshape(self.x_train.shape)


        #define pre-computations to make your life happier

        v_1 = (dhx_1**2+dhy_1**2)**(1/2)
        dhv_x_1 = (v_1)**(-1/2)*dhx_1
        dhv_y_1 = (v_1)**(-1/2)*dhy_1
        dcos_slip_1_x = torch.sin(torch.atan(dhy_1/dhx_1))*(dhy_1*dhx_1**-2)/(1+(dhy_1/dhx_1)**2)
        dcos_slip_1_y = -torch.sin(torch.atan(dhy_1/dhx_1))*(dhx_1**-1)/(1+(dhy_1/dhx_1)**2)
        cos_theta_1 = torch.cos(htheta_1)
        sin_theta_1 = torch.sin(htheta_1)
        tan_delta_1 = torch.tan(hdelta_1)
        cos_slip_1 = torch.cos(torch.atan(dhy_1/dhx_1))

        v_2 = (dhx_2**2+dhy_2**2)**(1/2)
        dhv_x_2 = (v_2)**(-1/2)*dhx_2
        dhv_y_2 = (v_2)**(-1/2)*dhy_2
        dcos_slip_2_x = torch.sin(torch.atan(dhy_2/dhx_2))*(dhy_2*dhx_2**-2)/(1+(dhy_2/dhx_2)**2)
        dcos_slip_2_y = -torch.sin(torch.atan(dhy_2/dhx_2))*(dhx_2**-1)/(1+(dhy_2/dhx_2)**2)
      
        
        cos_theta_2 = torch.cos(htheta_2)
        sin_theta_2 = torch.sin(htheta_2)
        tan_delta_2 = torch.tan(hdelta_2)
        cos_slip_2 = torch.cos(torch.atan(dhy_2/dhx_2))

        l_dx_1 = dhx_1 - v_1*cos_theta_1
        l_dy_1 = dhy_1 - v_1*sin_theta_1
        l_ddx_1 = ddhx_1+lambda_ddx_1
        l_ddy_1 = ddhy_1+lambda_ddy_1
        l_dtheta_1 = dhtheta_1 - (v_1*tan_delta_1*cos_slip_1/self.l)
        l_ddelta_1 = dhdelta_1+lambda_ddelta_1
        
        # l_lambda_dx_1 = d_lambda_dx_1 -(4*((hx_1-hx_2)**2-self.d**2)*(hx_1-hx_2))
        # l_lambda_dy_1 = d_lambda_dy_1 -(4*((hy_1-hy_2)**2-self.d**2)*(hy_1-hy_2))
        l_lambda_dx_1 = d_lambda_dx_1
        l_lambda_dy_1 = d_lambda_dy_1
        l_lambda_ddx_1 = d_lambda_ddx_1 -(-lambda_dx_1*cos_theta_1*dhv_x_1-lambda_dy_1*sin_theta_1*dhv_x_1-lambda_dtheta_1*tan_delta_1/self.l*(dhv_x_1*cos_slip_1+dcos_slip_1_x*v_1))
        l_lambda_ddy_1 = d_lambda_ddy_1 -(-lambda_dx_1*cos_theta_1*dhv_y_1-lambda_dy_1*sin_theta_1*dhv_y_1-lambda_dtheta_1*tan_delta_1/self.l*(dhv_y_1*cos_slip_1+dcos_slip_1_y*v_1))
        l_lambda_dtheta_1 = d_lambda_dtheta_1 - ( lambda_dx_1*v_1*sin_theta_1-lambda_dy_1*v_1*cos_theta_1)
        l_lambda_ddelta_1 = d_lambda_ddelta_1 - ( -lambda_dtheta_1/self.l*cos_slip_1*v_1*(1/torch.cos(hdelta_1))**2)


        l_dx_2 = dhx_2 - v_2*cos_theta_1
        l_dy_2 = dhy_2 - v_2*sin_theta_1
        l_ddx_2 = ddhx_2+lambda_ddx_2
        l_ddy_2 = ddhy_2+lambda_ddy_2
        l_dtheta_2 = dhtheta_2 - (v_2*tan_delta_2*cos_slip_2/self.l)
        l_ddelta_2 = dhdelta_2+lambda_ddelta_2
        
        
        # l_lambda_dx_2 = d_lambda_dx_2 +(4*((hx_1-hx_2)**2-self.d**2)*(hx_1-hx_2))
        # l_lambda_dy_2 = d_lambda_dy_2 +(4*((hy_1-hy_2)**2-self.d**2)*(hy_1-hy_2))
        l_lambda_dx_2 = d_lambda_dx_2
        l_lambda_dy_2 = d_lambda_dy_2
        l_lambda_ddx_2 = d_lambda_ddx_2 -(-lambda_dx_2*cos_theta_2*dhv_x_2-lambda_dy_2*sin_theta_2*dhv_x_2-lambda_dtheta_2*tan_delta_2/self.l*(dhv_x_2*cos_slip_2+dcos_slip_2_x*v_2))
        l_lambda_ddy_2 = d_lambda_ddy_2 -(-lambda_dx_2*cos_theta_2*dhv_y_2-lambda_dy_2*sin_theta_2*dhv_y_2-lambda_dtheta_2*tan_delta_2/self.l*(dhv_y_2*cos_slip_2+dcos_slip_2_y*v_2))
        l_lambda_dtheta_2 = d_lambda_dtheta_2 -(lambda_dx_2*v_2*sin_theta_2-lambda_dy_2*v_2*cos_theta_2)
        l_lambda_ddelta_2 = d_lambda_ddelta_2 -(-lambda_dtheta_2/self.l*cos_slip_2*v_2*(1/torch.cos(hdelta_2))**2)
        
        
        loss= torch.vstack((  
                              l_dx_1,
                              l_dy_1,
                              l_ddx_1,
                              l_ddy_1,
                              l_dtheta_1,
                              l_ddelta_1,
                              l_lambda_dx_1,
                              l_lambda_dy_1,
                              l_lambda_ddx_1,
                              l_lambda_ddy_1,
                              l_lambda_dtheta_1,
                              l_lambda_ddelta_1,
                              l_dx_2,
                              l_dy_2,
                              l_ddx_2,
                              l_ddy_2,
                              l_dtheta_2,
                              l_ddelta_2,
                              l_lambda_dx_2,
                              l_lambda_dy_2,
                              l_lambda_ddx_2,
                              l_lambda_ddy_2,
                              l_lambda_dtheta_2,
                              l_lambda_ddelta_2
                              ))  

        return loss

    def get_h(self,x):
        return torch.tanh(torch.add(torch.matmul(x,torch.transpose(self.W,0,1)),torch.transpose(self.b,0,1)))
    def get_dh(self,x):
        return torch.mul((1-self.get_h(x)**2),torch.transpose(self.W,0,1))
    def get_dhh(self,x):
        return -torch.mul((self.get_dh(x)),torch.transpose(self.W,0,1))
    
    def train(self,n_iterations,x_train,y_train_1,y_train_2,l=4.97,lambda_=1):
        
        
        count = 0
        
        self.lambda_ = lambda_
        
        z0 = -1
        zf = 1
        t0 = x_train[0]
        tf = x_train[-1]
        c = (zf-z0)/(tf-t0)
        x_train = z0+c*(x_train-t0)
        self.c = c
        
        self.x_train = torch.tensor(x_train,dtype=torch.float).reshape(x_train.shape[0],1)
        self.y_train_1 = torch.tensor(y_train_1,dtype=torch.float)
        
        self.y_train_2 = torch.tensor(y_train_2,dtype=torch.float)
       
        self.l = torch.tensor(l,dtype=torch.float)
        losses =((self.predict_loss(self.betas))**2).reshape(1,80,24) 
        print("number of samples:",len(self.x_train))
        while count < n_iterations:
            
            with torch.no_grad():
                
                jac = jacobian(self.predict_jacobian,self.betas)
                
                loss = self.predict_loss(self.betas)
                
                jac = jac.reshape(jac.shape[0],jac.shape[2])
                pinv_jac = torch.linalg.pinv(jac)
                delta = torch.matmul(pinv_jac,loss).reshape(self.betas.shape)
                self.betas -=delta*0.5
                print("final loss:",(loss**2).mean())
            if count>0:
                loss_e=(loss**2).reshape(1,80,24) 
                losses = torch.vstack((losses,loss_e))
                
            if count %10==0:
                
                print("final loss:",(loss**2).mean(),"10 step")
            
            count +=1

        # iters = losses.shape[1]
        
        losses = losses.mean(dim=1)
        print(losses.mean(dim=0))
        legends_states = [ "l_dx","l_dy","l_dxx","l_dyy",'l_dtheta','l_ddelta']
        legends_costates = [ "l_lambda_dx","l_lambda_dy","l_lambda_ddx","l_lambda_ddy","lambda_dtheta","lambda_ddelta"]
        # for j in range(10):
        #     if j < 5:
        #         print("states")
        
        # plt.figure()
        # plt.title("vehicle one losses_states")
        # plt.plot(losses[0,-10:].detach().numpy())
        # plt.plot(losses[1,-10:].detach().numpy())
        # plt.plot(losses[2,-10:].detach().numpy())
        # plt.plot(losses[3,-10:].detach().numpy())
        # plt.plot(losses[4,-10:].detach().numpy())
        # plt.plot(losses[5,-10:].detach().numpy())
        # plt.legend(legends_states)

        # plt.savefig("losses_states_veh1.png")
        # plt.figure()
        # plt.title("vehicle one losses_costates")
        # plt.plot(losses[6,-10:].detach().numpy())
        # plt.plot(losses[7,-10:].detach().numpy())
        # plt.plot(losses[8,-10:].detach().numpy())
        # plt.plot(losses[9,-10:].detach().numpy())
        # plt.plot(losses[10,-10:].detach().numpy())
        # plt.plot(losses[11,-10:].detach().numpy())
        # plt.legend(legends_costates)
        # plt.savefig("losses_costates_veh1.png")
        # plt.figure()
        # plt.title("vehicle two losses_states")
        # plt.plot(losses[0+12,-10:].detach().numpy())
        # plt.plot(losses[1+12,-10:].detach().numpy())
        # plt.plot(losses[2+12,-10:].detach().numpy())
        # plt.plot(losses[3+12,-10:].detach().numpy())
        # plt.plot(losses[4+12,-10:].detach().numpy())
        # plt.plot(losses[5+12,-10:].detach().numpy())
        # plt.legend(legends_states)
        # plt.savefig("losses_states_veh2.png")
        # plt.figure()
        # plt.title("vehicle two losses_costates")
        # plt.plot(losses[6+12,-10:].detach().numpy())
        # plt.plot(losses[7+12,-10:].detach().numpy())
        # plt.plot(losses[8+12,-10:].detach().numpy())
        # plt.plot(losses[9+12,-10:].detach().numpy())
        # plt.plot(losses[10+12,-10:].detach().numpy())
        # plt.plot(losses[11+12,-10:].detach().numpy())
        # plt.legend(legends_costates)
        # plt.savefig("losses_costates_veh2.png")
            
        
        
    def pred(self):

        
        betas = self.betas
        h = self.get_h(self.x_train)
        dh = self.get_dh(self.x_train)
        dhh = self.get_dhh(self.x_train)

        bx_1 = betas[0:self.nodes]
        by_1 = betas[self.nodes:self.nodes*2]
        btheta_1 = betas[self.nodes*2:self.nodes*3]
        bdelta_1 = betas[self.nodes*3:self.nodes*4]
        
        bx_2 = betas[self.nodes*10:self.nodes*11]
        by_2 = betas[self.nodes*11:self.nodes*12]
        btheta_2 = betas[self.nodes*12:self.nodes*13]
        bdelta_2 = betas[self.nodes*13:self.nodes*14]


        init_x_1 = self.y_train_1[0,0]        
        final_x_1 = self.y_train_1[-1,0]
        init_y_1 = self.y_train_1[0,1]
        final_y_1 = self.y_train_1[-1,1]
        init_theta_1 = self.y_train_1[0,2]        
        final_theta_1 = self.y_train_1[-1,2]
        init_vx_1 = self.y_train_1[0,3]        
        init_vy_1 = self.y_train_1[0,4]
        init_ax_1 = self.y_train_1[0,5]        
        init_ay_1 = self.y_train_1[0,6]
        init_h_1 = self.get_h(self.x_train[0])
        final_h_1 = self.get_h(self.x_train[-1])
        init_dh_1 = self.get_dh(self.x_train[0])
        init_dhh_1 = self.get_dhh(self.x_train[0])
        

        init_x_2 = self.y_train_2[0,0]        
        final_x_2 = self.y_train_2[-1,0]
        init_y_2 = self.y_train_2[0,1]
        final_y_2 = self.y_train_2[-1,1]
        init_theta_2 = self.y_train_2[0,2]        
        final_theta_2 = self.y_train_2[-1,2]
        init_vx_2 = self.y_train_2[0,3]        
        init_vy_2 = self.y_train_2[0,4]
        init_ax_2 = self.y_train_2[0,5]        
        init_ay_2 = self.y_train_2[0,6]

        cross_x_2 = self.y_train_2[1,0]
        cross_y_2 = self.y_train_2[1,1]
        r1_x_2 =self.y_train_2[2,0]
        r1_y_2 =self.y_train_2[2,1]
        r2_x_2 =self.y_train_2[3,0]
        r2_y_2 =self.y_train_2[3,1]
        r3_x_2 =self.y_train_2[4,0]
        r3_y_2 =self.y_train_2[4,1]

        rel_cross =  self.y_train_2[1,-1]
        rel_r1 = self.y_train_2[2,-1]
        rel_r2 = self.y_train_2[3,-1]
        rel_r3 = self.y_train_2[4,-1]

        final_time = self.x_train[-1].numpy()[0]
        init_time = self.x_train[0].numpy()[0]
        
        
        cross_time= ((final_time-init_time)*rel_cross+init_time).reshape(self.x_train[0].shape)
        r1_time= ((final_time-init_time)*rel_r1+init_time).reshape(self.x_train[0].shape)
        r2_time= ((final_time-init_time)*rel_r2+init_time).reshape(self.x_train[0].shape)
        r3_time= ((final_time-init_time)*rel_r3+init_time).reshape(self.x_train[0].shape)
        
        cross_index = np.where(self.x_train>=cross_time.numpy()[0])[0][0]
        cross_time = self.x_train[cross_index]
        r1_index = np.where(self.x_train>=r1_time.numpy()[0])[0][0]
        r1_time = self.x_train[r1_index]
        r2_index = np.where(self.x_train>=r2_time.numpy()[0])[0][0]
        r2_time = self.x_train[r2_index]
        r3_index = np.where(self.x_train>=r3_time.numpy()[0])[0][0]
        r3_time = self.x_train[r3_index]
        

        init_h_2 = self.get_h(self.x_train[0])
        cross_h_2 = self.get_h(cross_time)
        r1_h_2 = self.get_h(r1_time)
        r2_h_2 = self.get_h(r2_time)
        r3_h_2 = self.get_h(r3_time)
        final_h_2 = self.get_h(self.x_train[-1])
        init_dh_2 = self.get_dh(self.x_train[0])
        init_ddh_2 = self.get_dhh(self.x_train[0])
        cross_time = cross_time.numpy()[0]
        r1_time = r1_time.numpy()[0]
        r2_time = r2_time.numpy()[0]
        r3_time = r3_time.numpy()[0]
        
        
        # x_linspace[0:len(self.x_train)/2]=torch.linspace(init_x_2,final_x_2,int(abs(final_x_2.detach().numpy()-init_x_2.detach().numpy())/2))
        # x_linspace[len(self.x_train)/2:]= final_x_2
        # y_linspace[len(self.x_train)/2:]=torch.linspace(init_y_2,final_y_2,int(abs(final_y_2.detach().numpy()-init_y_2.detach().numpy())/2))
        # y_linspace[0:len(self.x_train)/2]= init_y_2
        # plt.figure()
        # plt.plot(x_linspace.detach().numpy(),plt.plot(y_linspace.detach().numpy()))
        # plt.show()
        """""
        For a 4 point constrained problem on x and y  we have: 

        Consider support functions the polynomials: t^0,t^1,t^2,t^3: Then for x and y we have:
        x(0)       [[1,0,0,0]
        x(f)       [1,t,t^2,t^3]
        xdot(0)    [0,1,0,0]
        xdotdot(0) [0,0,2,0]]

        Consider support functions the polynomials: t^0,t^1,t^2,t^3,t^4,t^5: Then for x and y we have:
        x(0)       [[1,t(0),t(0)^2,t(0)^3,t(0)^4,t(0)^5]
        x(r1)      [1,t(r1),t(r1)^2,t(r1)^3,t(r1)^4,t(r1)^5]
        x(r3)      [1,t(r3),t(r3)^2,t(r3)^3,t(r3)^4,t(r3)^5]
        x(f)       [1,t(rf),t(rf)^2,t(rf)^3]
        xdot(0)    [0,1,0,0]
        xdotdot(0) [0,0,2,0]]
        

        For a 2 point constrained problem on theta we have:
        Consider support functions the polynomials: t^0,t^1: Then for theta have:
        theta(0)       [[1,0]
        theta(f)       [1,t]
        """""
        
        support_function_matrix = np.array([[1,init_time,init_time**2,init_time**3],[1,final_time,final_time**2,final_time**3],[0,1,2*init_time,3*init_time**2],[0,0,2,6*init_time]])
        coefficients_matrix = torch.tensor(np.linalg.inv(support_function_matrix),dtype=torch.float)
        
        free_support_function_matrix = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train,self.x_train**2,self.x_train**3))
        d_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape),2*self.x_train,3*self.x_train**2))
        dd_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.zeros(size=self.x_train.shape),2*torch.ones(size=self.x_train.shape),6*self.x_train))

        
        support_function_matrix_2 = np.array([[ 1,init_time,init_time**2,init_time**3,init_time**4,init_time**5,init_time**6,init_time**7]\
                                              ,[1,final_time,final_time**2,final_time**3,final_time**4,final_time**5,final_time**6,final_time**7]\
                                              ,[1,cross_time,cross_time**2,cross_time**3,cross_time**4,cross_time**5,cross_time**6,cross_time**7]\
                                              ,[1,r1_time,r1_time**2,r1_time**3,r1_time**4,r1_time**5,r1_time**6,r1_time**7]\
                                              ,[1,r2_time,r2_time**2,r2_time**3,r2_time**4,r2_time**5,r2_time**6,r2_time**7]\
                                              ,[1,r3_time,r3_time**2,r3_time**3,r3_time**4,r3_time**5,r3_time**6,r3_time**7]\
                                              ,[0,1,2*init_time,3*init_time**2,4*init_time**3,5*init_time**4,6*init_time**5,7*init_time**6]\
                                              ,[0,0,2,6*init_time,12*init_time**2,20*init_time**3,30*init_time**4,42*init_time**5]])
        
        coefficients_matrix_2 = torch.tensor(np.linalg.inv(support_function_matrix_2),dtype=torch.float)

        free_support_function_matrix_2 = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train,self.x_train**2,self.x_train**3,self.x_train**4,self.x_train**5,self.x_train**6,self.x_train**7))
        
        d_free_support_function_matrix_2 = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape),2*self.x_train,3*self.x_train**2\
                                                         ,4*self.x_train**3,5*self.x_train**4,6*self.x_train**5,7*self.x_train**6))
        dd_free_support_function_matrix_2 = torch.hstack((torch.zeros(size=self.x_train.shape),torch.zeros(size=self.x_train.shape),2*torch.ones(size=self.x_train.shape)\
                                                          ,6*self.x_train,12*self.x_train**2,20*self.x_train**3,30*self.x_train**4,42*self.x_train**5))

        
        phis = torch.matmul(free_support_function_matrix,coefficients_matrix)
        phi1 = phis[:,0].reshape(len(self.x_train),1)
        phi2 = phis[:,1].reshape(len(self.x_train),1)
        phi3 = phis[:,2].reshape(len(self.x_train),1)
        phi4 = phis[:,3].reshape(len(self.x_train),1)
        d_phis = torch.matmul(d_free_support_function_matrix,coefficients_matrix)
        d_phi1 = d_phis[:,0].reshape(len(self.x_train),1)
        d_phi2 = d_phis[:,1].reshape(len(self.x_train),1)
        d_phi3 = d_phis[:,2].reshape(len(self.x_train),1)
        d_phi4 = d_phis[:,3].reshape(len(self.x_train),1)
        dd_phis = torch.matmul(dd_free_support_function_matrix,coefficients_matrix)
        dd_phi1 = dd_phis[:,0].reshape(len(self.x_train),1)
        dd_phi2 = dd_phis[:,1].reshape(len(self.x_train),1)
        dd_phi3 = dd_phis[:,2].reshape(len(self.x_train),1)
        dd_phi4 = dd_phis[:,3].reshape(len(self.x_train),1)

        
        phi1_h1 = torch.matmul(-phi1,init_h_1)
        phi1_x1_init = phi1*init_x_1
        phi1_y1_init = phi1*init_y_1
        phi2_hf = torch.matmul(-phi2,final_h_1)
        phi2_x1_final = phi2*final_x_1
        phi2_y1_final = phi2*final_y_1
        phi3_dh1 = torch.matmul(-phi3,init_dh_1)
        phi3_vx1_init = phi3*init_vx_1
        phi3_vy1_init = phi3*init_vy_1
        phi4_ddh1 = torch.matmul(-phi4,init_dhh_1)
        phi4_ax1_init = phi4*init_ax_1
        phi4_ay1_init = phi4*init_ay_1
        
        d_phi1_h1 = torch.matmul(-d_phi1,init_h_1)
        d_phi1_x1_init = d_phi1*init_x_1
        d_phi1_y1_init = d_phi1*init_y_1
        d_phi2_hf = torch.matmul(-d_phi2,final_h_1)
        d_phi2_x1_final = d_phi2*final_x_1
        d_phi2_y1_final = d_phi2*final_y_1
        d_phi3_dh1 = torch.matmul(-d_phi3,init_dh_1)
        d_phi3_vx1_init = d_phi3*init_vx_1
        d_phi3_vy1_init = d_phi3*init_vy_1
        d_phi4_ddh1 = torch.matmul(-d_phi4,init_dhh_1)
        d_phi4_ax1_init = d_phi4*init_ax_1
        d_phi4_ay1_init = d_phi4*init_ay_1
        
        dd_phi1_h1 = torch.matmul(-dd_phi1,init_h_1)
        dd_phi1_x1_init = dd_phi1*init_x_1
        dd_phi1_y1_init = dd_phi1*init_y_1
        dd_phi2_hf = torch.matmul(-dd_phi2,final_h_1)
        dd_phi2_x1_final = dd_phi2*final_x_1
        dd_phi2_y1_final = dd_phi2*final_y_1
        dd_phi3_dh1 = torch.matmul(-dd_phi3,init_dh_1)
        dd_phi3_vx1_init = dd_phi3*init_vx_1
        dd_phi3_vy1_init = dd_phi3*init_vy_1
        dd_phi4_ddh1 = torch.matmul(-dd_phi4,init_dhh_1)
        dd_phi4_ax1_init = dd_phi4*init_ax_1
        dd_phi4_ay1_init = dd_phi4*init_ay_1
        
   

        phis_2 = torch.matmul(free_support_function_matrix_2,coefficients_matrix_2)
        phi1_2 = phis_2[:,0].reshape(len(self.x_train),1)
        phi2_2 = phis_2[:,1].reshape(len(self.x_train),1)
        phi3_2 = phis_2[:,2].reshape(len(self.x_train),1)
        phi4_2 = phis_2[:,3].reshape(len(self.x_train),1)
        phi5_2 = phis_2[:,4].reshape(len(self.x_train),1)
        phi6_2 = phis_2[:,5].reshape(len(self.x_train),1)
        phi7_2 = phis_2[:,6].reshape(len(self.x_train),1)
        phi8_2 = phis_2[:,7].reshape(len(self.x_train),1)



        d_phis_2 = torch.matmul(d_free_support_function_matrix_2,coefficients_matrix_2)
        d_phi1_2 = d_phis_2[:,0].reshape(len(self.x_train),1)
        d_phi2_2 = d_phis_2[:,1].reshape(len(self.x_train),1)
        d_phi3_2 = d_phis_2[:,2].reshape(len(self.x_train),1)
        d_phi4_2 = d_phis_2[:,3].reshape(len(self.x_train),1)
        d_phi5_2 = d_phis_2[:,4].reshape(len(self.x_train),1)
        d_phi6_2 = d_phis_2[:,5].reshape(len(self.x_train),1)
        d_phi7_2 = d_phis_2[:,6].reshape(len(self.x_train),1)
        d_phi8_2 = d_phis_2[:,7].reshape(len(self.x_train),1)




        dd_phis_2 = torch.matmul(dd_free_support_function_matrix_2,coefficients_matrix_2)
        dd_phi1_2 = dd_phis_2[:,0].reshape(len(self.x_train),1)
        dd_phi2_2 = dd_phis_2[:,1].reshape(len(self.x_train),1)
        dd_phi3_2 = dd_phis_2[:,2].reshape(len(self.x_train),1)
        dd_phi4_2 = dd_phis_2[:,3].reshape(len(self.x_train),1)
        dd_phi5_2 = dd_phis_2[:,4].reshape(len(self.x_train),1)
        dd_phi6_2 = dd_phis_2[:,5].reshape(len(self.x_train),1)
        dd_phi7_2 = dd_phis_2[:,6].reshape(len(self.x_train),1)
        dd_phi8_2 = dd_phis_2[:,7].reshape(len(self.x_train),1)
   

        phi1_h1_2 = torch.matmul(-phi1_2,init_h_2)
        phi1_x2_init_2 = phi1_2*init_x_2
        phi1_y2_init_2 = phi1_2*init_y_2
        phi2_hf_2 = torch.matmul(-phi2_2,final_h_2)
        phi2_x2_final_2 = phi2_2*final_x_2
        phi2_y2_final_2 = phi2_2*final_y_2
        phi3_hcross_2 = torch.matmul(-phi3_2,cross_h_2)
        phi3_x2_cross_2 = phi3_2*cross_x_2
        phi3_y2_cross_2 = phi3_2*cross_y_2
        phi4_hr1_2 = torch.matmul(-phi4_2,r1_h_2)
        phi4_x2_r1_2 = phi4_2*r1_x_2
        phi4_y2_r1_2 = phi4_2*r1_y_2
        phi5_hr2_2 = torch.matmul(-phi5_2,r2_h_2)
        phi5_x2_r2_2 = phi5_2*r2_x_2
        phi5_y2_r2_2 = phi5_2*r2_y_2
        phi6_hr3_2 = torch.matmul(-phi6_2,r3_h_2)
        phi6_x2_r3_2 = phi6_2*r3_x_2
        phi6_y2_r3_2 = phi6_2*r3_y_2
        phi7_dh2_2 = torch.matmul(-phi7_2,init_dh_2)
        phi7_vx2_init_2 = phi7_2*init_vx_2
        phi7_vy2_init_2 = phi7_2*init_vy_2
        phi8_ddh2_2 = torch.matmul(-phi8_2,init_ddh_2)
        phi8_ax2_init_2 = phi8_2*init_ax_2
        phi8_ay2_init_2 = phi8_2*init_ay_2
        
        d_phi1_h1_2 = torch.matmul(-d_phi1_2,init_h_2)
        d_phi1_x2_init_2 = d_phi1_2*init_x_2
        d_phi1_y2_init_2 = d_phi1_2*init_y_2
        d_phi2_hf_2 = torch.matmul(-d_phi2_2,final_h_2)
        d_phi2_x2_final_2 = d_phi2_2*final_x_2
        d_phi2_y2_final_2 = d_phi2_2*final_y_2
        d_phi3_hcross_2 = torch.matmul(-d_phi3_2,cross_h_2)
        d_phi3_x2_cross_2 = d_phi3_2*cross_x_2
        d_phi3_y2_cross_2 = d_phi3_2*cross_y_2
        d_phi4_hr1_2 = torch.matmul(-d_phi4_2,r1_h_2)
        d_phi4_x2_r1_2 = d_phi4_2*r1_x_2
        d_phi4_y2_r1_2 = d_phi4_2*r1_y_2
        d_phi5_hr2_2 = torch.matmul(-d_phi5_2,r2_h_2)
        d_phi5_x2_r2_2 = d_phi5_2*r2_x_2
        d_phi5_y2_r2_2 = d_phi5_2*r2_y_2
        d_phi6_hr3_2 = torch.matmul(-d_phi6_2,r3_h_2)
        d_phi6_x2_r3_2 = d_phi6_2*r3_x_2
        d_phi6_y2_r3_2 = d_phi6_2*r3_y_2
        d_phi7_dh2_2 = torch.matmul(-d_phi7_2,init_dh_2)
        d_phi7_vx2_init_2 = d_phi7_2*init_vx_2
        d_phi7_vy2_init_2 = d_phi7_2*init_vy_2
        d_phi8_ddh2_2 = torch.matmul(-d_phi8_2,init_ddh_2)
        d_phi8_ax2_init_2 = d_phi8_2*init_ax_2
        d_phi8_ay2_init_2 = d_phi8_2*init_ay_2


        dd_phi1_h1_2 = torch.matmul(-dd_phi1_2,init_h_2)
        dd_phi1_x2_init_2 = dd_phi1_2*init_x_2
        dd_phi1_y2_init_2 = dd_phi1_2*init_y_2
        dd_phi2_hf_2 = torch.matmul(-dd_phi2_2,final_h_2)
        dd_phi2_x2_final_2 = dd_phi2_2*final_x_2
        dd_phi2_y2_final_2 = dd_phi2_2*final_y_2
        dd_phi3_hcross_2 = torch.matmul(-dd_phi3_2,cross_h_2)
        dd_phi3_x2_cross_2 = dd_phi3_2*cross_x_2
        dd_phi3_y2_cross_2 = dd_phi3_2*cross_y_2
        dd_phi4_hr1_2 = torch.matmul(-dd_phi4_2,r1_h_2)
        dd_phi4_x2_r1_2 = dd_phi4_2*r1_x_2
        dd_phi4_y2_r1_2 = dd_phi4_2*r1_y_2
        dd_phi5_hr2_2 = torch.matmul(-dd_phi5_2,r2_h_2)
        dd_phi5_x2_r2_2 = dd_phi5_2*r2_x_2
        dd_phi5_y2_r2_2 = dd_phi5_2*r2_y_2
        dd_phi6_hr3_2 = torch.matmul(-dd_phi6_2,r3_h_2)
        dd_phi6_x2_r3_2 = dd_phi6_2*r3_x_2
        dd_phi6_y2_r3_2 = dd_phi6_2*r3_y_2
        dd_phi7_dh2_2 = torch.matmul(-dd_phi7_2,init_dh_2)
        dd_phi7_vx2_init_2 = dd_phi7_2*init_vx_2
        dd_phi7_vy2_init_2 = dd_phi7_2*init_vy_2
        dd_phi8_ddh2_2 = torch.matmul(-dd_phi8_2,init_ddh_2)
        dd_phi8_ax2_init_2 = dd_phi8_2*init_ax_2
        dd_phi8_ay2_init_2 = dd_phi8_2*init_ay_2



        
        hx_1 = torch.matmul(h.add(phi1_h1).add(phi2_hf).add(phi3_dh1).add(phi4_ddh1),bx_1).reshape(self.x_train.shape)\
            .add(phi1_x1_init).add(phi2_x1_final).add(phi3_vx1_init/self.c).add(phi4_ax1_init/self.c**2)
        
        dhx_1 = self.c*(torch.matmul(dh.add(d_phi1_h1).add(d_phi2_hf).add(d_phi3_dh1).add(d_phi4_ddh1),bx_1).reshape(self.x_train.shape)\
            .add(d_phi1_x1_init).add(d_phi2_x1_final).add(d_phi3_vx1_init/self.c).add(d_phi4_ax1_init/self.c**2))
        
        ddhx_1 = self.c**2*(torch.matmul(dhh.add(dd_phi1_h1).add(dd_phi2_hf).add(dd_phi3_dh1).add(dd_phi4_ddh1),bx_1).reshape(self.x_train.shape)\
            .add(dd_phi1_x1_init).add(dd_phi2_x1_final).add(dd_phi3_vx1_init/self.c).add(dd_phi4_ax1_init/self.c**2))
        
        hy_1 = torch.matmul(h.add(phi1_h1).add(phi2_hf).add(phi3_dh1).add(phi4_ddh1),by_1).reshape(self.x_train.shape)\
            .add(phi1_y1_init).add(phi2_y1_final).add(phi3_vy1_init/self.c).add(phi4_ay1_init/self.c**2)
        
        dhy_1 = self.c*(torch.matmul(dh.add(d_phi1_h1).add(d_phi2_hf).add(d_phi3_dh1).add(d_phi4_ddh1),by_1).reshape(self.x_train.shape)\
            .add(d_phi1_y1_init).add(d_phi2_y1_final).add(d_phi3_vy1_init/self.c).add(d_phi4_ay1_init/self.c**2))
        
        ddhy_1 = self.c**2*(torch.matmul(dhh.add(dd_phi1_h1).add(dd_phi2_hf).add(dd_phi3_dh1).add(dd_phi4_ddh1),by_1).reshape(self.x_train.shape)\
            .add(dd_phi1_y1_init).add(dd_phi2_y1_final).add(dd_phi3_vy1_init/self.c).add(dd_phi4_ay1_init/self.c**2))
        


        hx_2 = torch.matmul(h.add(phi1_h1_2).add(phi2_hf_2).add(phi3_hcross_2).add(phi4_hr1_2).add(phi5_hr2_2).add(phi6_hr3_2).add(phi7_dh2_2).add(phi8_ddh2_2),bx_2).reshape(self.x_train.shape)\
            .add(phi1_x2_init_2).add(phi2_x2_final_2).add(phi3_x2_cross_2).add(phi4_x2_r1_2).add(phi5_x2_r2_2).add(phi6_x2_r3_2).add(phi7_vx2_init_2/self.c).add(phi8_ax2_init_2/self.c**2)
        
        dhx_2 = self.c*(torch.matmul(dh.add(d_phi1_h1_2).add(d_phi2_hf_2).add(d_phi3_hcross_2).add(d_phi4_hr1_2).add(d_phi5_hr2_2).add(d_phi6_hr3_2).add(d_phi7_dh2_2).add(d_phi8_ddh2_2),bx_2).reshape(self.x_train.shape)\
            .add(d_phi1_x2_init_2).add(d_phi2_x2_final_2).add(d_phi3_x2_cross_2).add(d_phi4_x2_r1_2).add(d_phi5_x2_r2_2).add(d_phi6_x2_r3_2).add(d_phi7_vx2_init_2/self.c).add(d_phi8_ax2_init_2/self.c**2))

        ddhx_2 = self.c**2*(torch.matmul(dhh.add(dd_phi1_h1_2).add(dd_phi2_hf_2).add(dd_phi3_hcross_2).add(dd_phi4_hr1_2).add(dd_phi5_hr2_2).add(dd_phi6_hr3_2).add(dd_phi7_dh2_2).add(dd_phi8_ddh2_2),bx_2).reshape(self.x_train.shape)\
            .add(dd_phi1_x2_init_2).add(dd_phi2_x2_final_2).add(dd_phi3_x2_cross_2).add(dd_phi4_x2_r1_2).add(dd_phi5_x2_r2_2).add(dd_phi6_x2_r3_2).add(dd_phi7_vx2_init_2/self.c).add(dd_phi8_ax2_init_2/self.c**2))



        hy_2 = torch.matmul(h.add(phi1_h1_2).add(phi2_hf_2).add(phi3_hcross_2).add(phi4_hr1_2).add(phi5_hr2_2).add(phi6_hr3_2).add(phi7_dh2_2).add(phi8_ddh2_2),by_2).reshape(self.x_train.shape)\
            .add(phi1_y2_init_2).add(phi2_y2_final_2).add(phi3_y2_cross_2).add(phi4_y2_r1_2).add(phi5_y2_r2_2).add(phi6_y2_r3_2).add(phi7_vy2_init_2/self.c).add(phi8_ay2_init_2/self.c**2)
        
        dhy_2 = self.c*(torch.matmul(dh.add(d_phi1_h1_2).add(d_phi2_hf_2).add(d_phi3_hcross_2).add(d_phi4_hr1_2).add(d_phi5_hr2_2).add(d_phi6_hr3_2).add(d_phi7_dh2_2).add(d_phi8_ddh2_2),by_2).reshape(self.x_train.shape)\
            .add(d_phi1_y2_init_2).add(d_phi2_y2_final_2).add(d_phi3_y2_cross_2).add(d_phi4_y2_r1_2).add(d_phi5_y2_r2_2).add(d_phi6_y2_r3_2).add(d_phi7_vy2_init_2/self.c).add(d_phi8_ay2_init_2/self.c**2))

        ddhy_2 = self.c**2*(torch.matmul(dhh.add(dd_phi1_h1_2).add(dd_phi2_hf_2).add(dd_phi3_hcross_2).add(dd_phi4_hr1_2).add(dd_phi5_hr2_2).add(dd_phi6_hr3_2).add(dd_phi7_dh2_2).add(dd_phi8_ddh2_2),by_2).reshape(self.x_train.shape)\
            .add(dd_phi1_y2_init_2).add(dd_phi2_y2_final_2).add(dd_phi3_y2_cross_2).add(dd_phi4_y2_r1_2).add(dd_phi5_y2_r2_2).add(dd_phi6_y2_r3_2).add(dd_phi7_vy2_init_2/self.c).add(dd_phi8_ay2_init_2/self.c**2))




        y_y_1 = hy_1
        y_dy_1 = dhy_1
        y_ddy_1 = ddhy_1
        y_x_1 = hx_1
        y_dx_1 = dhx_1
        y_ddx_1 = ddhx_1
        y_y_2 = hy_2
        y_dy_2 = dhy_2
        y_ddy_2 = ddhy_2
        y_x_2 = hx_2
        y_dx_2 = dhx_2
        y_ddx_2 = ddhx_2
        heavy = lambda x,k: 1/2 +1/2*torch.tanh(k*x)
        x_c = 68
        y_c = 590.5
        x_left_conflict = 80
        #Add inequalities for x and y for car 1

        
        min_dist =  y_x_2-y_x_1+y_y_2-y_y_1
        heaviside_min_dist = heavy(self.d-min_dist,self.k)
        heaviside_r1 = heavy(x_c-y_x_1,self.k)
        heaviside_r2 = heavy(y_y_2-y_c,self.k)
        heaviside_r3 = heavy(y_x_2-x_c,self.k)
        heaviside_r4 = heavy(x_left_conflict-y_x_2,self.k)

        region1 = heaviside_min_dist*heaviside_r1
        region2 = heaviside_min_dist*heaviside_r2
        region3 = heaviside_r3*heaviside_r4
        

        m = (r2_y_2-r1_y_2)/(r2_x_2-r1_x_2)
        intercept = r1_y_2-m*r1_x_2
        # hx_1 = y_x_1 + (-(self.d+4.9)+y_x_2+y_y_2-y_y_1-y_x_1)*(region1)
        # dhx_1 = y_dx_1 + (y_dx_2+y_dy_2-y_dy_1-y_dx_1)*(region1)
        # ddhx_1 = y_ddx_1 + (y_ddx_2+y_ddy_2-y_ddy_1-y_ddx_1)*(region1)

        # hy_1 = y_y_1 + (-(self.d+4.9)+y_x_2-y_x_1+y_y_2-y_y_1)*(region1)
        # dhy_1 = y_dy_1 + (y_dx_2-y_dx_1+y_dy_2-y_dy_1)*(region1)
        # ddhy_1 = y_ddy_1 + (y_ddx_2-y_ddx_1+y_ddy_2-y_ddy_1)*(region1)
       
       

        path_x = hy_2/m-intercept
        dpath_x = dhy_2/m
        ddpath_x = ddhy_2/2
        path_y = hx_2*m+intercept
        dpath_y = dhx_2*m
        ddpath_y = ddhx_2*m

        
        hx_2 = hx_2 + (path_x-hx_2)*region3
        dhx_2 = dhx_2 + (dpath_x-dhx_2)*(region3)
        ddhx_2 = ddhx_2 + (ddpath_x-ddhx_2)*(region3)
      
        hy_2 = hy_2 + (path_y-hy_2)*region3
        dhy_2 = dhy_2 + (dpath_y-dhy_2)*(region3)
        ddhy_2 = ddhy_2 + (ddpath_y-ddhy_2)*(region3)

        plt.figure()
        plt.scatter(hx_2.detach().numpy(),hy_2.detach().numpy())
        # plt.plot(hx_2.detach().numpy(),path_y.detach().numpy())
        plt.show()

        hx_2 = y_x_2 + ((self.d+4.9)+y_x_1-y_y_2+y_y_1-y_x_2)*(region2)
        dhx_2 = y_dx_2 + (y_dx_1-y_dy_2+y_dy_1-y_dx_2)*(region2)
        ddhx_2 = y_ddx_2 + (y_ddx_1-y_ddy_2+y_ddy_1-y_ddx_2)*(region2)
      
        hy_2 = y_y_2 + ((self.d+4.9)+y_x_1-y_x_2+y_y_1-y_y_2)*(region2)
        dhy_2 = y_dy_2 + (y_dx_1-y_dx_2+y_dy_1-y_dy_2)*(region2)
        ddhy_2 = y_ddy_2 + (y_dx_1-y_dx_2+y_dy_1-y_dy_2)*(region2)


        f_r_u_1 = init_y_1+self.d
        f_r_l_1 = init_y_1-self.d
        heaviside_r_u = heavy(y_y_1-f_r_u_1,self.k)
        heaviside_r_l =  heavy(f_r_l_1-y_y_1,self.k)

        y_y_1 = hy_1
        y_dy_1 = dhy_1
        y_ddy_1 = ddhy_1


        hy_1 = y_y_1 +(f_r_u_1-y_y_1)*(heaviside_r_u)+(f_r_l_1-y_y_1)*(heaviside_r_l)
        dhy_1 = y_dy_1 +(-y_dy_1)*(heaviside_r_u)+(-y_dy_1)*(heaviside_r_l)
        ddhy_1 = y_ddy_1 +(-y_ddy_1)*(heaviside_r_u)-(y_ddy_1)*(heaviside_r_l)

        plt.figure()
        plt.scatter(hx_1.detach().numpy(),hy_1.detach().numpy())
        # plt.scatter(hx_1[indices_1].detach().numpy(),hy_1[indices_1].detach().numpy(),c="red")
        plt.scatter(hx_2.detach().numpy(),hy_2.detach().numpy())
        # plt.scatter(hx_2[indices_2].detach().numpy(),hy_2[indices_2].detach().numpy(),c="red")
        plt.show()

       
        


        support_function_matrix = np.array([[1,init_time],[1,final_time]])
        coefficients_matrix = torch.tensor(np.linalg.inv(support_function_matrix),dtype=torch.float)
        
        free_support_function_matrix = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train))
        d_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape)))
        
        phis = torch.matmul(free_support_function_matrix,coefficients_matrix)
        phi1 = phis[:,0].reshape(len(self.x_train),1)
        phi2 = phis[:,1].reshape(len(self.x_train),1)
        d_phis = torch.matmul(d_free_support_function_matrix,coefficients_matrix)
        d_phi1 = d_phis[:,0].reshape(len(self.x_train),1)
        d_phi2 = d_phis[:,1].reshape(len(self.x_train),1)
        
        
        phi1_theta1_init = phi1*init_theta_1
        phi1_theta2_init = phi1*init_theta_2
        phi2_theta1_final = phi2*final_theta_1
        phi2_theta2_final = phi2*final_theta_2
       
        d_phi1_theta1_init = d_phi1*init_theta_1
        d_phi1_theta2_init = d_phi1*init_theta_2
        d_phi2_theta1_final = d_phi2*final_theta_1
        d_phi2_theta2_final = d_phi2*final_theta_2


        htheta_1 = torch.matmul(h.add(phi1_h1).add(phi2_hf),btheta_1).reshape(self.x_train.shape).add(phi1_theta1_init).add(phi2_theta1_final)
        dhtheta_1 = self.c*torch.matmul(dh.add(d_phi1_h1).add(d_phi2_hf),btheta_1).reshape(self.x_train.shape).add(d_phi1_theta1_init).add(d_phi2_theta1_final)

        hdelta_1 = torch.matmul(h,bdelta_1).reshape(self.x_train.shape)
        dhdelta_1 = torch.matmul(dh,bdelta_1).reshape(self.x_train.shape)

        htheta_2 = torch.matmul(h.add(phi1_h1).add(phi2_hf),btheta_2).reshape(self.x_train.shape).add(phi1_theta2_init).add(phi2_theta2_final)
        dhtheta_2 = self.c*torch.matmul(dh.add(d_phi1_h1).add(d_phi2_hf),btheta_2).reshape(self.x_train.shape).add(d_phi1_theta2_init).add(d_phi2_theta2_final)

        hdelta_2 = torch.matmul(h,bdelta_2).reshape(self.x_train.shape)
        dhdelta_2 = torch.matmul(dh,bdelta_2).reshape(self.x_train.shape)

        
        return torch.hstack((hx_1,
                             hy_1,
                             htheta_1,
                             hdelta_1,
                             dhx_1,
                             dhy_1,
                             dhtheta_1,
                             dhdelta_1,
                             ddhx_1,
                             ddhy_1,
                             hx_2,
                             hy_2,
                             htheta_2,
                             hdelta_2,
                             dhx_2,
                             dhy_2,
                             dhtheta_2,
                             dhdelta_2,
                             ddhx_2,
                             ddhy_2))
                             
                             
                             
                             
                             