import pickle 
import os
import pandas as pd
import numpy as np
# from task1 import train2, recover, load_data_
# from utils import *
import torch

from retro_star.api import RSPlanner

planner = RSPlanner(
    gpu=-1,
    use_value_fn=True,
    iterations=100,
    expansion_topk=50
)


# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def load_data(dir):
    path = f"./data/Multi-Step task/{dir}.pkl"
    data = pd.read_pickle(path)
    return data

starting = load_data('starting_mols')
test = load_data('test_mols')
# routes = load_data('target_mol_route')

print('load over!')
# print(test)
# print(starting)

# knn = train2(k = 20, metric='cosine')
# MLP = ValueMLP(device=device, n_layers=3)
# n=5
# train=load_data_('train')
# X_s = train[0]
# y_s = train[1]

# def predict(product, score):
#     index = knn.kneighbors([product], n)[1]
#     print(index)
#     print(index.shape)
#     templates = y_s[index.tolist()]
#     print(templates)
#     for t in templates:
#         reactants = recover(t, product)
#         print(reactants)
#         re = reactants.split('.')

# predict()

pred_routes = []
for s in test:
    print(s)
    result = planner.plan(s)
    print(result)
    pred_routes.append(result['routes'])

np.save('routes.npy', pred_routes)











