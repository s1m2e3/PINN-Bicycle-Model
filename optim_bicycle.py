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
        self.d = d

    def predict_jacobian(self,betas):

        init_x_1 = self.y_train[0,0]        
        init_hx_1 = 
        
        final_x_1 = self.y_train[-1,0]
        final_hx_1 = 

        init_y_1 = self.y_train[0,1]        
        final_y_1 = self.y_train[-1,1]

        init_theta_1 = self.y_train[0,2]        
        final_theta_1 = self.y_train[-1,2]

        init_dx_1 = self.y_train[0,3]        
        init_dhx_1 = 

        init_dy_1 = self.y_train[0,4]

        init_ddelta_1 = self.y_train[0,5]        
        
        switch_1 = 

        
        hx_1 = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[0:self.nodes])\
            + self.y_train[0,0]
        hy_1 = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes:self.nodes*2])\
              + self.y_train[0,1]
        htheta_1 = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*2:self.nodes*3])\
              + self.y_train[0,2]
        hdelta_1 = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*3:self.nodes*4])\
              + self.y_train[0,3]
        
        lambda_x_1=torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*4:self.nodes*5])\
              + self.y_train[0,3]
        
        lambda_y_1=torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*5:self.nodes*6])\
              + self.y_train[0,3]
        lambda_dx_1=torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*6:self.nodes*7])\
              + self.y_train[0,3]
        
        lambda_dy_1=torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*7:self.nodes*8])\
              + self.y_train[0,3]
        lambda_theta_1 = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*8:self.nodes*9])\
              + self.y_train[0,3]
        lambda_delta_2 = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*9:self.nodes*10])\
              + self.y_train[0,3]



        
        dhx_1 =  self.c*torch.matmul(self.get_dh(self.x_train),betas[0:self.nodes])
        dhxx_1 = self.c**2*torch.matmul(self.get_dhh(self.x_train),betas[0:self.nodes])
        dhy_1 = self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes:2*self.nodes])
        dhyy_1 = self.c**2*torch.matmul(self.get_dhh(self.x_train),betas[self.nodes:2*self.nodes])
        dhtheta_1 =  self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes*2:3*self.nodes])
        dhdelta_1 = self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes*3:4*self.nodes])


        hx_2 = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*4:5*self.nodes])\
            + self.y_train[0,0]
        hy_2 = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*5:6*self.nodes])\
              + self.y_train[0,1]
        htheta_2 = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*6:7*self.nodes])\
              + self.y_train[0,2]
        hdelta_2 = torch.matmul(torch.add(self.get_h(self.x_train_pred),-self.get_h(self.x_train_pred[0])),betas[self.nodes*7:8*self.nodes])\
              + self.y_train[0,3]
        
        dhx_2 =  self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes*4:5*self.nodes])
        dhxx_2 = self.c**2*torch.matmul(self.get_dhh(self.x_train),betas[self.nodes*4:5*self.nodes])
        dhy_2 = self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes*5:6*self.nodes])
        dhyy_2 = self.c**2*torch.matmul(self.get_dhh(self.x_train),betas[self.nodes*5:6*self.nodes])
        dhtheta_2 =  self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes*6:7*self.nodes])
        dhdelta_2 = self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes*7:8*self.nodes])

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