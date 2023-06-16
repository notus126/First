import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

input_dim = 2048
n_layers=1
fp_dim=2048
latent_dim=128


class MLP(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.fc4 = nn.Softmax(dim=1)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        tensor = self.fc4(tensor)
        return tensor

class ValueMLP(nn.Module):
    def __init__(self, device, n_layers=2, fp_dim=2048, latent_dim=128, dropout_rate=0.1):
        super(ValueMLP, self).__init__()
        self.n_layers = n_layers
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.device = device


        layers = []
        layers.append(nn.Linear(fp_dim, latent_dim))
        # layers.append(nn.BatchNorm1d(latent_dim,
        #                              track_running_stats=False))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            # layers.append(nn.BatchNorm1d(latent_dim,
            #                              track_running_stats=False))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(latent_dim, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, fps):
        x = fps
        x = self.layers(x)
        x = torch.log(1 + torch.exp(x))
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 256, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        
        # Apply the first convolutional layer
        x = x.unsqueeze(1)  # Add a channel dimension to the input tensor
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Apply the second convolutional layer
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten the output of the second convolutional layer and pass it through the fully connected layers
        x=x.view(-1,self.num_flat_features(x))
    
        # Apply the first fully connected layer
        print(x.size())
        x=self.fc1(x)
        x=self.relu3(x)
    
        # Apply the second fully connected layer and return the output
        x=self.fc2(x)
        return x.squeeze()

    def num_flat_features(self,x):
        size=x.size()[1:] #All dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        # print(self.len)
        return self.len
        # pass
    
    def get(self):
        return self.X, self.y