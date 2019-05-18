# -*- coding: utf-8 -*-
"""

@author: Marguerite
"""

import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture


from sklearn.cluster import KMeans

import pandas as pd
from sklearn import datasets
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model


# Load actual dataset 
# Please place apprio. dataset here. 
# pro_data_1 = pd.read_csv("left_turn_mag.csv")


flt_cols = ['acc_x_blp', 'acc_y_blp', 'acc_z_blp', 'gyr_x_blp', 'gyr_y_blp', 'gyr_z_blp', 'mag_x_blp', 'mag_y_blp', 'mag_z_blp', 'type']
 
def train_model(pro_data):
    #load baseline data
    models = []
    gauss_mix_model = mixture.GaussianMixture(n_components=9)
    kmeans = KMeans(5, random_state=0)
    regr_model = linear_model.LinearRegression()

    # Extract features from dataset. 
    X = pro_data[['acc_x_blp', 'acc_y_blp', 'acc_z_blp', 'gyr_x_blp', 'gyr_y_blp', 'gyr_z_blp', 'mag_x_blp', 'mag_y_blp', 'mag_z_blp']] 
    Y = pro_data['type']

    # Apply linear regression to data. 
    regr_model.fit(X,Y)
    gauss_mix_model.fit(X)
    kmeans.fit(X)

    models.append(gauss_mix_model)
    models.append(kmeans)
    models.append(regr_model)
    return models

def get_max_freq(test_list):
   res = test_list[0] 
   max = 0
   for i in test_list: 
        freq = test_list.count(i) 
        if freq > max: 
            max = freq 
            res = i 
   return max, res

def get_predictions(dval, models):
    n_mdls = len(models)
    labels_pred = {}
    prob_pred = {}
    dval = dval[['acc_x_blp', 'acc_y_blp', 'acc_z_blp', 'gyr_x_blp', 'gyr_y_blp', 'gyr_z_blp', 'mag_x_blp', 'mag_y_blp', 'mag_z_blp']]
    
    nrows = dval.shape[0]
    for i in range(n_mdls):
        labels_pred[i] = models[i].predict(dval)
        #try:
           #prob_pred[i] = models[i].predict_proba(dval)
        #except Exception:
        assume_prob = []
        for x in range(nrows):
           assume_prob.append(0.65)
        prob_pred[i] = assume_prob
        #print(prob_pred[i])
    out_lbl = []
    prob = []
    for i in range(nrows):
        lbl_p = []
        for m in range(n_mdls):
             #print(prob_pred[m][i])
             #print(labels_pred[m][i])
             if prob_pred[m][i] > .60:
                  lbl_p.append(labels_pred[m][i])
        m_l, r_l = get_max_freq(lbl_p)
        out_lbl.append(r_l)
        prob.append((m_l/n_mdls))      
    return out_lbl, prob
    
