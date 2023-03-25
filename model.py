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
        self.fc1 = nn.Linear(input_size1, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def train(self,num_epochs,x_train_data,y_train_data):
        for epoch in range(num_epochs):
            for i in range(len(x_train_data)):
                
                x = x_train_data[i,:,]
                y = y_train_data[i,:,]
                self.optimizer.zero_grad()
                outputs = self(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                # Print training statistics
                if (i+1) % 10 == 0:
                    print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                        .format(epoch+1, num_epochs, i+1, len(x_train_data), loss.item()))


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.Relu()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out = self.relu(out)
        
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def train(self,num_epochs,x_train_data,y_train_data):
        
        for epoch in range(num_epochs):
            for i in range(len(x_train_data)):
                
                x = x_train_data[i,:,]
                y = y_train_data[i,:,]
                self.optimizer.zero_grad()
                outputs = self(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                # Print training statistics
                if (i+1) % 10 == 0:
                    print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                        .format(epoch+1, num_epochs, i+1, len(x_train_data), loss.item()))