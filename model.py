import torch
import torch.nn as nn

def reshape_tensor(x, batch_size):
    original_shape = x.size()
    num_elements = x.numel()
    new_shape = (batch_size,) + original_shape[1:]
    if num_elements != new_shape[0] * torch.tensor(new_shape[1:]).prod():
        raise ValueError("Number of elements in tensor does not match new shape")
    return x.view(new_shape)

class NN(nn.Module):
    def __init__(self, input_size1, hidden_size, output_size):
        super(NN, self).__init__()
        print(input_size1)
        self.fc1 = nn.Linear(input_size1, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.optimizer = torch.optim.SGD(self.parameters(),lr=0.1)

    def forward(self, x):
        x=torch.tensor(x,dtype=torch.float)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def train(self,num_epochs,x_train_data,y_train_data):
        x_train_data = torch.tensor(x_train_data,dtype=torch.float)
        y_train_data = torch.tensor(y_train_data,dtype=torch.float)
        for epoch in range(num_epochs):
            for i in range(len(x_train_data)):
                
                #x = x_train_data[i,:,]
                #y = y_train_data[i,:,]
                self.optimizer.zero_grad()
                outputs = self.forward(x_train_data)
                self.criterion = nn.MSELoss()
                loss = self.criterion(outputs, y_train_data)
                loss.backward()
                self.optimizer.step()
                
                # Print training statistics
                if (i+1) % 10 == 0:
                    print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                        .format(epoch+1, num_epochs, i+1, len(x_train_data), loss.item()))


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,input_sequence_length,output_sequence_length):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size,hidden_size, num_layers,batch_first=True,dtype=torch.double)
        self.relu = nn.ReLU()
        self.output_sequence_length = output_sequence_length
        self.input_sequence_length = input_sequence_length
        self.fc1 = nn.Linear(hidden_size, hidden_size,dtype=torch.double)
        self.fc2 = nn.Linear(hidden_size, output_size,dtype=torch.double)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(),lr=0.1)


    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,dtype=torch.double).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,dtype=torch.double).to(x.device)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        out = self.relu(out[:,-self.output_sequence_length:,:])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out=out[0:out.shape[0]-self.output_sequence_length:,:,:]
        return out

    def train(self,num_epochs,x_train_data,y_train_data):
        x_train_data = torch.tensor(x_train_data,dtype=torch.double)
        y_train_data = torch.tensor(y_train_data,dtype=torch.double)
        
        for epoch in range(num_epochs):
        
            self.optimizer.zero_grad()
            outputs = self.forward(x_train_data)
            
            
            loss = self.criterion(outputs, y_train_data[self.input_sequence_length:,:])
            loss.backward()
            self.optimizer.step()
            
            # Print training statistics
            if (epoch+1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                    .format(epoch+1, num_epochs, epoch+1, len(x_train_data), loss.item()))
