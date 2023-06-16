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


f=open("./task2_1.log")
line = f.readline().strip()
Loss=[]
while line:
    loss = line.split(',')[1].split(': ')[1]
    # print(loss)
    Loss.append(float(loss))
    line=f.readline().strip()

plt.plot(np.arange(1, len(Loss)+1), Loss)
# plt.save('task2.png')
plt.savefig('task2.png', dpi=300)
