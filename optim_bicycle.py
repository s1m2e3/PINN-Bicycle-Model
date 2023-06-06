from bicycle_PINN import *

class XTFC_veh(PIELM):
    def __init__(self,n_nodes,input_size,output_size,length,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh",d=3):
        super().__init__(n_nodes,input_size,output_size,length,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh",d=3)

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

    def predict_jacobian(self,betas):
        

        init_x_1 = self.y_train_1[0,0]        
        final_x_1 = self.y_train_1[-1,0]
        init_y_1 = self.y_train_1[0,1]
        final_y_1 = self.y_train_1[-1,-1]
        init_theta_1 = self.y_train_1[0,2]        
        final_theta_1 = self.y_train_1[-1,2]
        init_vx_1 = self.y_train_1[0,3]        
        init_vy_1 = self.y_train_1[0,4]
        init_ax_1 = self.y_train_1[0,5]        
        init_ay_1 = self.y_train_1[0,6]

        init_h_1 = self.get_h(self.x_train_pred[0])
        final_h_1 = self.get_h(self.x_train_pred[-1])
        init_dh_1 = self.get_dh(self.x_train_pred[0])
        init_dhh_1 = self.get_dhh(self.x_train_pred[0])

        init_x_2 = self.y_train_2[0,0]        
        final_x_2 = self.y_train_2[-1,0]
        init_y_2 = self.y_train_2[0,1]
        final_y_2 = self.y_train_2[-1,-1]
        init_theta_2 = self.y_train_2[0,2]        
        final_theta_2 = self.y_train_2[-1,2]
        init_vx_2 = self.y_train_2[0,3]        
        init_vy_2 = self.y_train_2[0,4]
        init_ax_2 = self.y_train_2[0,5]        
        init_ay_2 = self.y_train_2[0,6]
     
        init_h_2 = self.get_h(self.x_train_pred[0])
        final_h_2 = self.get_h(self.x_train_pred[-1])
        init_dh_2 = self.get_dh(self.x_train_pred[0])
        init_dhh_2 = self.get_dhh(self.x_train_pred[0])

        
        
        """""
        For a 4 point constrained problem on x and y  we have: 

        Consider support functions the polynomials: t^0,t^1,t^2,t^3: Then for x and y we have:
        x(0)       [[1,0,0,0]
        x(f)       [1,t,t^2,t^3]
        xdot(0)    [0,1,0,0]
        xdotdot(0) [0,0,2,0]]

        For a 2 point constrained problem on theta we have:
        Consider support functions the polynomials: t^0,t^1: Then for theta have:
        theta(0)       [[1,0]
        theta(f)       [1,t]
        """""
        
        final_time = 10
        support_function_matrix = np.array([[1,0,0,0],[0,final_time,final_time**2,final_time**3],[0,1,0,0],[0,0,2,0]])
        coefficients_matrix = np.linalg.inv(support_function_matrix)
        
        free_support_function_matrix = np.array([np.ones(len(self.x_train_pred)),self.x_train_pred,self.x_train_pred**2,self.x_train_pred**3]).T
        d_free_support_function_matrix = np.array([np.zeros(len(self.x_train_pred)),np.ones(len(self.x_train_pred)),2*self.x_train_pred,3*self.x_train_pred**2]).T
        dd_free_support_function_matrix = np.array([np.zeros(len(self.x_train_pred)),np.zeros(len(self.x_train_pred)),2*np.ones(len(self.x_train_pred)),6*self.x_train_pred]).T

        phis = np.diagonal(np.dot(free_support_function_matrix,coefficients_matrix))
        d_phis = np.diagonal(np.dot(d_free_support_function_matrix,coefficients_matrix))
        dd_phis = np.diagonal(np.dot(dd_free_support_function_matrix,coefficients_matrix))
        
        hx_1 = torch.matmul(self.get_h(self.x_train_pred).add(-phis[0]*init_h_1).add(-phis[1]*final_h_1).add(-phis[2]*init_dh_1).add(-phis[3]*init_dhh_1),betas[0:self.nodes])\
            .add(phis[0]*init_x_1).add(phis[1]*final_x_1).add(phis[2]*init_vx_1/self.c).add(phis[3]*init_ax_1/self.c**2)
        dhx_1 = self.c*torch.matmul(self.get_dh(self.x_train_pred).add(-d_phis[0]*init_h_1).add(-d_phis[1]*final_h_1).add(-d_phis[2]*init_dh_1).add(-d_phis[3]*init_dhh_1),betas[0:self.nodes])\
            .add(d_phis[0]*init_x_1).add(d_phis[1]*final_x_1).add(d_phis[2]*init_vx_1/self.c).add(d_phis[3]*init_ax_1/self.c**2)
        ddhx_1 = self.c**2*torch.matmul(self.get_dhh(self.x_train_pred).add(-dd_phis[0]*init_h_1).add(-dd_phis[1]*final_h_1).add(-dd_phis[2]*init_dh_1).add(-dd_phis[3]*init_dhh_1),betas[0:self.nodes])\
            .add(dd_phis[0]*init_x_1).add(dd_phis[1]*final_x_1).add(dd_phis[2]*init_vx_1/self.c).add(dd_phis[3]*init_ax_1/self.c**2)
        
        hy_1 = torch.matmul(self.get_h(self.x_train_pred).add(-phis[0]*init_h_1).add(-phis[1]*final_h_1).add(-phis[2]*init_dh_1).add(-phis[3]*init_dhh_1),betas[self.nodes:self.nodes*2])\
            .add(phis[0]*init_y_1).add(phis[1]*final_y_1).add(phis[2]*init_vy_1/self.c).add(phis[3]*init_ay_1/self.c**2)
        dhy_1 = self.c*torch.matmul(self.get_dh(self.x_train_pred).add(-d_phis[0]*init_h_1).add(-d_phis[1]*final_h_1).add(-d_phis[2]*init_dh_1).add(-d_phis[3]*init_dhh_1),betas[self.nodes:self.nodes*2])\
            .add(d_phis[0]*init_y_1).add(d_phis[1]*final_y_1).add(d_phis[2]*init_vy_1/self.c).add(d_phis[3]*init_ay_1/self.c**2)
        ddhy_1 = self.c**2*torch.matmul(self.get_dhh(self.x_train_pred).add(-dd_phis[0]*init_h_1).add(-dd_phis[1]*final_h_1).add(-dd_phis[2]*init_dh_1).add(-dd_phis[3]*init_dhh_1),betas[self.nodes:self.nodes*2])\
            .add(dd_phis[0]*init_y_1).add(dd_phis[1]*final_y_1).add(dd_phis[2]*init_vy_1/self.c).add(dd_phis[3]*init_ay_1/self.c**2)
      
        hx_2 = torch.matmul(self.get_h(self.x_train_pred).add(-phis[0]*init_h_2).add(-phis[1]*final_h_2).add(-phis[2]*init_dh_2).add(-phis[3]*init_dhh_2),betas[self.nodes*10:self.nodes*11])\
            .add(phis[0]*init_x_2).add(phis[2]*final_x_2).add(phis[2]*init_vx_2/self.c).add(phis[3]*init_ax_2/self.c**2)
        dhx_2 = self.c*torch.matmul(self.get_dh(self.x_train_pred).add(-d_phis[0]*init_h_2).add(-d_phis[1]*final_h_2).add(-d_phis[2]*init_dh_2).add(-d_phis[3]*init_dhh_2),betas[self.nodes*10:self.nodes*11])\
            .add(d_phis[0]*init_x_2).add(d_phis[1]*final_x_2).add(d_phis[2]*init_vx_2/self.c).add(d_phis[3]*init_ax_2/self.c**2)
        ddhx_2 = self.c**2*torch.matmul(self.get_dhh(self.x_train_pred).add(-dd_phis[0]*init_h_2).add(-dd_phis[1]*final_h_2).add(-dd_phis[2]*init_dh_2).add(-dd_phis[3]*init_dhh_2),betas[self.nodes*10:self.nodes*11])\
            .add(dd_phis[0]*init_x_2).add(dd_phis[1]*final_x_2).add(dd_phis[2]*init_vx_2/self.c).add(dd_phis[3]*init_ax_2/self.c**2)
        
        hy_2 = torch.matmul(self.get_h(self.x_train_pred).add(-phis[0]*init_h_2).add(-phis[1]*final_h_2).add(-phis[2]*init_dh_2).add(-phis[3]*init_dhh_2),betas[self.nodes*11:self.nodes*12])\
            .add(phis[0]*init_y_2).add(phis[1]*final_y_2).add(phis[2]*init_vy_2/self.c).add(phis[3]*init_ay_2/self.c**2)
        dhy_2 = self.c*torch.matmul(self.get_dh(self.x_train_pred).add(-d_phis[0]*init_h_2).add(-d_phis[1]*final_h_2).add(-d_phis[2]*init_dh_2).add(-d_phis[3]*init_dhh_2),betas[self.nodes*11:self.nodes*12])\
            .add(d_phis[0]*init_y_2).add(d_phis[1]*final_y_2).add(d_phis[2]*init_vy_2/self.c).add(d_phis[3]*init_ay_2/self.c**2)
        ddhy_2 = self.c**2*torch.matmul(self.get_dhh(self.x_train_pred).add(-dd_phis[0]*init_h_2).add(-dd_phis[1]*final_h_2).add(-dd_phis[2]*init_dh_2).add(-dd_phis[3]*init_dhh_2),betas[self.nodes*11:self.nodes*12])\
            .add(dd_phis[0]*init_y_2).add(dd_phis[1]*final_y_2).add(dd_phis[2]*init_vy_2/self.c).add(dd_phis[3]*init_ay_2/self.c**2)


        support_function_matrix = np.array([[1,0],[1,final_time]])
        coefficients_matrix = np.linalg.inv(support_function_matrix)
        
        free_support_function_matrix = np.array([np.ones(len(self.x_train_pred)),self.x_train_pred]).T
        d_free_support_function_matrix = np.array([np.zeros(len(self.x_train_pred)),np.ones(len(self.x_train_pred))]).T
        dd_free_support_function_matrix = np.array([np.zeros(len(self.x_train_pred)),np.zeros(len(self.x_train_pred))]).T

        phis = np.diagonal(np.dot(free_support_function_matrix,coefficients_matrix))
        d_phis = np.diagonal(np.dot(d_free_support_function_matrix,coefficients_matrix))
        dd_phis = np.diagonal(np.dot(dd_free_support_function_matrix,coefficients_matrix))

        htheta_1 = torch.matmul(self.get_h(self.x_train_pred).add(phis[0]*init_h_1).add(phis[1]*final_h_1),betas[self.nodes*2:self.nodes*3])\
              .add(phis[0]*init_theta_1).add(phis[1]*final_theta_1)
        dhtheta_1 = self.c*torch.matmul(self.get_dh(self.x_train_pred).add(d_phis[0]*init_h_1).add(d_phis[1]*final_h_1),betas[self.nodes*2:self.nodes*3])\
              .add(d_phis[0]*init_theta_1).add(d_phis[1]*final_theta_1) 
        hdelta_1 = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*3:self.nodes*4])
        dhdelta_1 = torch.matmul(self.get_dh(self.x_train_pred),betas[self.nodes*3:self.nodes*4])

        htheta_2 = torch.matmul(self.get_h(self.x_train_pred).add(phis[0]*init_h_2).add(phis[1]*final_h_2),betas[self.nodes*12:self.nodes*13])\
              .add(phis[0]*init_theta_2).add(phis[1]*final_theta_2)
        dhtheta_2 = self.c*torch.matmul(self.get_dh(self.x_train_pred).add(d_phis[0]*init_h_2).add(d_phis[1]*final_h_2),betas[self.nodes*12:self.nodes*13])\
              .add(d_phis[0]*init_theta_2).add(d_phis[1]*final_theta_2) 
        hdelta_2 = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*13:self.nodes*14])
        dhdelta_2 = torch.matmul(self.get_dh(self.x_train_pred),betas[self.nodes*13:self.nodes*14])
        

        lambda_x_1 = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*4:self.nodes*5])
        lambda_y_1 = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*5:self.nodes*6])
        lambda_dx_1 = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*6:self.nodes*7])
        lambda_dy_1= torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*7:self.nodes*8])
        lambda_dtheta_1 = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*8:self.nodes*9])
        lambda_ddelta_1 = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*9:self.nodes*10])

        lambda_x_2 = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*14:self.nodes*15])
        lambda_y_2 = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*15:self.nodes*16])
        lambda_dx_2 = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*16:self.nodes*17])
        lambda_dy_2= torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*17:self.nodes*18])
        lambda_dtheta_2 = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*18:self.nodes*19])
        lambda_ddelta_2 = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*19:self.nodes*20])

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

        l_dx_1 = dhx_1 - v_1*sin_theta_1
        l_dy_1 = dhy_1 - v_1*cos_theta_1
        l_ddx_1 = ddhx_1+lambda_dx_1
        l_ddy_1 = ddhy_1+lambda_dy_1
        l_dtheta_1 = dhtheta_1 - (v_1*tan_delta_1*cos_slip_1/self.l)
        l_ddelta_1 = dhdelta_1+lambda_ddelta_1

        l_dx_2 = dhx_2 - v_2*sin_theta_1
        l_dy_2 = dhy_2 - v_2*cos_theta_1
        l_ddx_2 = ddhx_2+lambda_dx_2
        l_ddy_2 = ddhy_2+lambda_dy_2
        l_dtheta_2 = dhtheta_2 - (v_2*tan_delta_2*cos_slip_2/self.l)
        l_ddelta_2 = dhdelta_2+lambda_ddelta_2

        l_lambda_dx_1 = -4((hx_1-hx_2)**2-self.d**2)*(hx_1-hx_2)
        l_lambda_dy_1 = -4((hy_1-hy_2)**2-self.d**2)*(hy_1-hy_2)

        
        l_lambda_ddx_1 = -lambda_dx_1*sin_theta_1*dhv_x_1-lambda_dy_1*cos_theta_1*dhv_x_1-lambda_dtheta_1*tan_delta_1/self.l*(dhv_x_1*cos_slip_1+dcos_slip_1_x*v_1)
        l_lambda_ddy_1 = -lambda_dx_1*sin_theta_1*dhv_y_1-lambda_dy_1*cos_theta_1*dhv_y_1-lambda_dtheta_1*tan_delta_1/self.l*(dhv_y_1*cos_slip_1+dcos_slip_1_y*v_1)
        l_lambda_dtheta_1 = -lambda_dx_1*v_1*cos_theta_1+lambda_dy_1*v_1*sin_theta_1
        l_lambda_ddelta_1 = -lambda_dtheta_1/self.l*cos_slip_1*v_1*(1/torch.cos(hdelta_1))**2
        l_lambda_dx_2 = 4((hx_1-hx_2)**2-self.d**2)*(hx_1-hx_2)
        l_lambda_dy_2 = 4((hy_1-hy_2)**2-self.d**2)*(hy_1-hy_2)
        l_lambda_ddx_2 = -lambda_dx_2*sin_theta_2*dhv_x_2-lambda_dy_2*cos_theta_2*dhv_x_2-lambda_dtheta_2*tan_delta_2/self.l*(dhv_x_2*cos_slip_2+dcos_slip_2_x*v_2)
        l_lambda_ddy_2 = -lambda_dx_2*sin_theta_2*dhv_y_2-lambda_dy_2*cos_theta_2*dhv_y_2-lambda_dtheta_2*tan_delta_2/self.l*(dhv_y_2*cos_slip_2+dcos_slip_2_y*v_2)
        l_lambda_dtheta_2 = -lambda_dx_2*v_2*cos_theta_2+lambda_dy_2*v_2*sin_theta_2
        l_lambda_ddelta_2 = -lambda_dtheta_2/self.l*cos_slip_2*v_2*(1/torch.cos(hdelta_2))**2
            





        l_lambda_x_1 = -4((hx_1-hx_2)**2-self.d**2)*(hx_1-hx_2)
        l_lambda_y_1 = -4((hy_1-hy_2)**2-self.d**2)*(hy_1-hy_2)
        l_lambda_dx_1 = 

        # htheta_full = torch.matmul(torch.add(self.get_h(self.x_train),-self.get_h(self.x_train_pred[0])),betas[self.nodes*2:self.nodes*3])\
            #   + self.y_train[0,2]
        # hdelta = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*3:self.nodes*4]) + self.y_train[0,3]
        # l_pred_x = self.y_train_pred[:,0]-hx
        # l_pred_y = self.y_train_pred[:,1]-hy
        # l_pred_theta = self.y_train_pred[:,2]-htheta
        # l_pred_delta = self.y_train[:,3]-hdelta
        
        l_lambda_x=-
        l_lambda_dx=
        
        l_lambda_y=
        l_lambda_dy=

        l_lambda_theta=
        l_lambda_delta=
        



        l_x = dhx-(((dhx)**2+ (dhy)**2)**(1/2)*torch.cos(htheta_full+self.slip_angle))
        l_y = dhy-(((dhx)**2+ (dhy)**2)**(1/2)*torch.sin(htheta_full+self.slip_angle)) 
        l_theta = dhtheta - (((dhx)**2+ (dhy)**2)**(1/2))*torch.tan(self.steering_angle)*torch.cos(self.slip_angle)/self.l
        # l_delta = dhdelta-self.rho
        l_pred_dhx = self.speed_x - dhx
        l_pred_dhy = self.speed_y - dhy
        l_pred_dhtheta = self.heading_ratio - dhtheta 
        #loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        # loss= torch.hstack((l_pred_x,l_pred_y,l_x,l_y))
        # loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                            # (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))
        # loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                            # self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,self.lambda_*l_pred_dhtheta,\
                            # (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))  
        loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                            self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,\
                            (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))  


        return loss
            
    def predict_loss(self,x,y,x_pred):
         
        hx = torch.matmul(torch.add(self.get_h(x_pred),-self.get_h(x_pred[0])),self.betas[0:self.nodes])+ self.y_train[0,0]
        hy = torch.matmul(torch.add(self.get_h(x_pred),-self.get_h(x_pred[0])),self.betas[self.nodes:self.nodes*2]) + self.y_train[0,1]
        htheta = torch.matmul(torch.add(self.get_h(x_pred),-self.get_h(x_pred[0])),self.betas[self.nodes*2:self.nodes*3]) + self.y_train[0,2]
        htheta_full = torch.matmul(torch.add(self.get_h(x),-self.get_h(x_pred[0])),self.betas[self.nodes*2:self.nodes*3]) + self.y_train[0,2]
        #hdelta = torch.matmul(torch.add(self.get_h(x_pred),-self.get_h(x_pred[0])),self.betas[self.nodes*3:self.nodes*4]) + self.y_train[0,3]
        
        
        l_pred_x = y[:,0]-hx
        l_pred_y = y[:,1]-hy
        l_pred_theta = y[:,2]-htheta
        # l_pred_delta = y[:,3]-hdelta


        dhx = self.c* torch.matmul(self.get_dh(x),self.betas[0:self.nodes])
        dhy = self.c*torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])
        dhtheta = self.c* torch.matmul(self.get_dh(x),self.betas[self.nodes*2:3*self.nodes])
        # dhdelta = torch.matmul(self.get_dh(x),self.betas[self.nodes*3:4*self.nodes])

        l_x = dhx-(((dhx)**2+ (dhy)**2)**(1/2)*torch.cos(self.y_train[:,2]))
        l_y = dhy-(((dhx)**2+ (dhy)**2)**(1/2)*torch.sin(self.y_train[:,2])) 
        l_theta = dhtheta - (((dhx)**2+ (dhy)**2)**(1/2))*torch.tan(self.steering_angle)*torch.cos(self.slip_angle)/self.l
        # l_delta = dhdelta-self.rho
        
        l_pred_dhx = self.speed_x- dhx
        l_pred_dhy = self.speed_y- dhy
        l_pred_dhtheta = self.heading_ratio - dhtheta 
        #loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        # loss= torch.hstack((l_pred_x,l_pred_y,l_x,l_y))
        # loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                            # (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta)) 
        # loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                            # self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,self.lambda_*l_pred_dhtheta,\
                            # (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))
        loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                            self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,\
                            (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))  
  
        return loss




        # x0 = -torch.matmul(self.get_h(x_pred),self.betas[0:self.nodes])[0]+y[0,0]
        # y0 = -torch.matmul(self.get_h(x_pred),self.betas[self.nodes:2*self.nodes])[0]+y[0,1]
        # theta0 = -torch.matmul(self.get_h(x_pred),self.betas[self.nodes*2:3*self.nodes])[0]+y[0,2]
        # delta0 =-torch.matmul(self.get_h(x_pred),self.betas[self.nodes*3:4*self.nodes])[0]+y[0,3]

        # hx = torch.matmul(self.get_h(x_pred),self.betas[0:self.nodes]) + x0
        # hy = torch.matmul(self.get_h(x_pred),self.betas[self.nodes:2*self.nodes]) +y0
        # htheta = torch.matmul(self.get_h(x_pred),self.betas[self.nodes*2:3*self.nodes]) +theta0
        # hdelta = torch.matmul(self.get_h(x_pred),self.betas[self.nodes*3:4*self.nodes]) +delta0

        # l_pred_x = y[:,0]-hx
        # l_pred_y = y[:,1]-hy
        # l_pred_theta = y[:,2]-htheta
        # l_pred_delta = y[:,3]-hdelta
        
        # l_x = torch.matmul(self.get_dh(x),self.betas[0:self.nodes])-\
        # (torch.matmul(self.get_dh(x),self.betas[0:self.nodes])**2+\
        # torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])**2)**(1/2)\
        # *torch.cos(torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes])+theta0)
        
        # l_y = torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])-\
        # (torch.matmul(self.get_dh(x),self.betas[0:self.nodes])**2+\
        # torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])**2)**(1/2)\
        # *torch.sin(torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes])+theta0) 
        
        # l_theta = torch.matmul(self.get_dh(x),self.betas[self.nodes*2:3*self.nodes])-\
        # (torch.matmul(self.get_dh(x),self.betas[0:self.nodes])**2\
        # +torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])**2)**(1/2)\
        # *torch.tan(torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes])+delta0)/self.l
        
        # l_delta = torch.matmul(self.get_dh(x),self.betas[self.nodes*3:4*self.nodes])-self.rho
        # loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        
        # return loss

    def get_h(self,x):
        return torch.tanh(torch.add(torch.matmul(x,torch.transpose(self.W,0,1)),torch.transpose(self.b,0,1)))
    def get_dh(self,x):
        return torch.mul((1-self.get_h(x)**2),torch.transpose(self.W,0,1))
    def get_dhh(self,x):
        return -torch.mul((self.get_dh(x)),torch.transpose(self.W,0,1))
    
    def pred(self,x):


        z0 = -1
        zf = 1
        t0 = x[0]
        tf = x[-1]
        c = (zf-z0)/(tf-t0)
        x = z0+c*(x-t0)
        
        x = torch.tensor(np.array(x),dtype=torch.float).reshape(x.shape[0],1)
        print(min(x))
        print(max(x))
        hx = torch.matmul(torch.add(self.get_h(x),-self.get_h(x[0])),self.betas[0:self.nodes])+ self.y_train[0,0]
        hy = torch.matmul(torch.add(self.get_h(x),-self.get_h(x[0])),self.betas[self.nodes:self.nodes*2]) + self.y_train[0,1]
        htheta = torch.matmul(torch.add(self.get_h(x),-self.get_h(x[0])),self.betas[self.nodes*2:self.nodes*3]) + self.y_train[0,2]
        delta_pred = torch.matmul(self.get_h(x),self.betas[self.nodes*3:4*self.nodes])  
        
        return torch.vstack((hx,hy,htheta,delta_pred))