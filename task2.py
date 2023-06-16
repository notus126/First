import pandas as pd
import numpy as np
from utils import *
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale



def load_data(dir):
    path = f"./data/MoleculeEvaluationData/{dir}.pkl"
    data = pd.read_pickle(path)
    return data

data = load_data('train')
X_s = np.unpackbits(data['packed_fp'], axis=1)
y_s = data['values']
X_s = scale(X_s)


data = load_data('test')
X_t = np.unpackbits(data['packed_fp'], axis=1)
y_t = data['values']
X_t = scale(X_t)



device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def accuracy(model, X_t=X_t, y_t=y_t, batch=500):

    X_t = torch.tensor(X_t, dtype=torch.float32).to(device)
    y_t = y_t.to(device)
    
    model.eval()
    # preds = model(X_t)
    criterion = nn.MSELoss()
    # loss = criterion(preds, y_t)
    test_set = MyDataset(X_t, y_t)
    test_loader = DataLoader(dataset=test_set, batch_size=batch, shuffle=True, num_workers=0)
    sum_loss = 0.0
    sum_R2 = 0.0
    sum = 0
    for i, (X,y) in enumerate(test_loader):
        preds = model(X)
        # print(((preds.detach().cpu().numpy().squeeze()-y.detach().cpu().numpy().squeeze()))/y.detach().cpu().numpy().squeeze())
        loss = criterion(preds, y)
        sum_loss += loss.item()
        sum_R2 += r2_score(y.detach().cpu().numpy().tolist(), preds.detach().cpu().numpy().tolist())
        sum += len(X)
    sum_loss /= sum
    sum_R2 /= sum
    return sum_loss, sum_R2

def train(epoch, lr, batch, X_s=X_s, y_s=y_s):
    print('start training!')
    # model = CNN(len(standards))
    # model = SVC(kernel='rbf')
    model = ValueMLP(dropout_rate=0.1, device=device, n_layers=3)
    # model = ConvNet()
    model = model.to(device)
    X_s = torch.tensor(X_s, dtype=torch.float32).to(device)
    # X_s = X_s.unsqueeze(1)
    y_s = y_s.to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion =  nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_set = MyDataset(X_s, y_s)
    train_loader = DataLoader(dataset=train_set, batch_size=batch, shuffle=True, num_workers=0)
    model.train()
    best_R2=-10000000000000000
    print('start training!')
    for e in range(epoch):
        sum_loss = 0.0
        sum_R2 = 0.0
        sum = 0
        for i, (X,y) in enumerate(train_loader):
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            sum_loss += loss.item()
            sum_R2 += r2_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy())
            sum += len(X)

            optimizer.step()
            
        sum_loss /= sum
        sum_R2 /= sum
        acc, R2 = accuracy(model)
        if R2>best_R2:
            torch.save(model.state_dict(), 'task2_model.pth')
            best_R2 = R2
        print(f'epoch: {e+1}, train_loss: {sum_loss}, test_loss: {acc}, train_R2: {sum_R2}, test_R2: {R2}')

# train(1000, 0.001, 1000)





