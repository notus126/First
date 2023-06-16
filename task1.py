from rdchiral.template_extractor import extract_from_reaction
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdchiral.main import rdchiralRunText
from torch.utils.data import Dataset, DataLoader


import os
import pandas as pd
import math

def load_data(data):
    path = os.getcwd()
    dir = path+f"/data/schneider50k/raw_{data}.csv"
    df = pd.read_csv(dir, encoding="utf-8")
    df_array = np.array(df)
    ID = df_array[:,0]
    Class = df_array[:,1]
    reactions = df_array[:,2]
    # print('load data finish')
    return ID, Class, reactions


def get_template(reaction):
    reactants, products = reaction.split('>>')
    inputRec = {'_id': None, 'reactants': reactants, 'products': products}
    ans = extract_from_reaction(inputRec)
    # print(ans)
    if 'reaction_smarts' in ans.keys():
        return ans['reaction_smarts']
    else:
        return None

def transformer(product):
    mol = Chem.MolFromSmiles(product)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits())
    arr[onbits] = 1
    return arr

def recover(template, product):
    out = rdchiralRunText(template, product)
    return out


# 除去数据集中不在训练集中出现的template
def stripe(train, val, test):
    # find={}
    T = np.unique(train[1])
    # for t in T:
    #     find[t]=True
    for i in range(len(test)):
        if not test[1][i] in T:
            del test[0][i]
            del test[1][i]
    
    for i in range(len(val)):
        if not val[1][i] in T:
            del val[0][i]
            del val[1][i]
    return val, test
    
# 提取数据为(Morgan FingerPrint, template)
def preprocess(reactions):
    data = []
    label = []
    for reaction in reactions:
        template = get_template(reaction)
        if template == None:
            continue
        reactants, products = reaction.split('>>')
        # d = (transformer(products), template)
        data.append(transformer(products))
        label.append(template)
    return (data,label)

# 初始化
def reshape():
    ID, Class, reactions1 = load_data('train')
    ID, Class, reactions2 = load_data('val')
    ID, Class, reactions3 = load_data('test')
    train = preprocess(reactions1)
    val = preprocess(reactions2)
    test = preprocess(reactions3)
    stripe_val, stripe_test = stripe(train, val, test)
    np.save('./data/schneider50k/train.npy', train)
    np.save('./data/schneider50k/val.npy', val)
    np.save('./data/schneider50k/test.npy', test)
    np.savez('./data/schneider50k/stripe_val_test.npz', val=stripe_val, test=stripe_test)

def load_stripe_data():
    data = np.load('./data/schneider50k/stripe_val_test.npz', allow_pickle=True)
    # print('load stripe data finish')
    val = data['val'].tolist()
    test = data['test'].tolist()
    for i in range(len(val[0])):
        val[0][i]=val[0][i].tolist()
    for i in range(len(test[0])):
        test[0][i]=test[0][i].tolist()

    return val, test

def load_data_(data):
    path = os.getcwd()
    dir = path+f"/data/schneider50k/{data}.npy"
    data = np.load(dir, allow_pickle=True)
    data = data.tolist()
    for i in range(len(data[0])):
        data[0][i]=data[0][i].tolist()
    return data

# reshape()
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
from sklearn.preprocessing import scale



def accuracy(model, dev, standards, type='stripe', sel='val'):
    if type == 'stripe':
        val, test = load_stripe_data()
        if sel=='val':
            data = val
        else:
            data = test
    else:
        # ID, Class, reactions = load_data(sel)
        # data = preprocess(reactions)
        data = load_data_(sel)

    X_t = torch.tensor(data[0],dtype=torch.float32).to(dev)
    y_t = data[1]
    
    model.eval()
    preds = model(X_t)
    preds = np.argmax(preds.detach().cpu().numpy().tolist(),axis=1)
    cor = 0
    for i in range(len(preds)):
        if standards[preds[i]]==y_t[i]:
            cor+=1
    acc = cor/len(y_t)
    return acc
    
    


def predict(knn, metric, n):
    train=load_data_('train')
    X_s = train[0]
    y_s = train[1]

    val, test = load_stripe_data()

    neighbors = knn.kneighbors(val[0], n)
    distances = neighbors[0]
    nodes = neighbors[1]
    cor1 = 0.0
    cor2 = 0.0
    for i in range(len(nodes)):
        kn = nodes[i]
        distance=distances[i]
        find_index = -1
        sum = 0
        for j in range(len(kn)):
            sum += math.exp(-distance[j]*distance[j])
            if y_s[kn[j]]==val[1][i]:
                find_index = j
                break
        if find_index>-1:
            cor1+=1
            cor2 += math.exp(-distance[find_index]*distance[find_index])/sum
    # print(f'metric: {metric}, k: {k}, n: {n}, accuracy: {cor1/len(val[1])}/{cor2/len(val[1])}')
    print(f'metric: {metric}, n: {n}, accuracy: {cor1/len(val[1])}/{cor2/len(val[1])}')


def train2(k=22, metric='minkowski'):
    train=load_data_('train')
    # X_s = scale(train[0])
    X_s = train[0]
    y_s = train[1]
    
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_s, y_s)
    return knn

def predict_template(knn, product, n):
    train=load_data_('train')
    y_s = train[1]
    neighbors = knn.kneighbors(product, n)
    distances = neighbors[0]
    nodes = neighbors[1]
    T = []
    P = []
    for i in range(len(nodes)):
        kn = nodes[i]
        distance=distances[i]
        prob = []
        ts = []
        sum = 0.0
        for j in range(n):
            ts.append(y_s[kn[j]])
            prob.append(distance[j])
            sum += math.exp(-distance[j]*distance[j])
        for j in range(n):
            prob[j]=math.exp(-prob[j]*prob[j])/sum
        T.append(ts)
        P.append(prob)
    return T, P



# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# print(device)

# for metric in ['cosine', 'mincowski', 'manhattan']:
#         knn = train2(20,'cosine')
#         for n in range(1,41):
#             predict(knn, metric, n)

knn = train2(20, 'cosine')
val, test = load_stripe_data()
X_t = test[0][10:100]
y_t = test[1][10:100]
T, P = predict_template(knn, X_t, 5)
for i in range(len(X_t)):
    print(f'True template: \n{y_t[i]}')
    ts = T[i]
    prob = P[i]
    for j in range(len(ts)):
        print(f'{ts[j]}: {prob[j]}')
    













