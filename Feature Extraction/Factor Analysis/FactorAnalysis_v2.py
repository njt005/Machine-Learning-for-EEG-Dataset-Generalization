#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
June 15, 2020
Nick Tacca
"""

import os
import numpy as np
from scipy.io import loadmat, savemat
from factor_analyzer import FactorAnalyzer, calculate_kmo
import pickle
import matplotlib.pyplot as plt

N_FACTORS = 111

def load_data(dataset):
    os.chdir('/home/nick/Documents/Hiwi/Data/Subject Data/Ehrlich 2016')
    if dataset == "TRAIN":
        data = loadmat(os.path.join(os.getcwd(), "Training_Data.mat"))
    elif dataset == "TEST":
        data = loadmat(os.path.join(os.getcwd(), "Test_Data.mat"))
    feats = data["feats"]
    labels = data["labels"]
    return feats, labels

def feature_extraction():
    # Load data
    x_train, y_train = load_data("TRAIN")
    x_test, y_test = load_data("TEST")

    # Downsample data and reshape for FA
    x_train = x_train[(11, 12, 31, 32, 33, 34, 46, 47, 48, 49), :, :]
    x_test = x_test[(11, 12, 31, 32, 33, 34, 46, 47, 48, 49), :, :]
    x_train = x_train.reshape((10*51, x_train.shape[2])).T
    x_test = x_test.reshape((10*51, x_test.shape[2])).T

    # Calculate KMO criteria -- if > 0.5, then suitable for FA
    kmo_all_train, kmo_model_train = calculate_kmo(x_train)
    kmo_all_test, kmo_model_test = calculate_kmo(x_test)

    valid_train = False
    if kmo_model_train > 0.5:
        valid_train = True

    valid_test = False
    if kmo_model_test > 0.5:
        valid_test = True

    print("KMO Train: {:.2f}  Valid: {}".format(kmo_model_train, valid_train))
    print("KMO Test: {:.2f}  Valid: {}".format(kmo_model_test, valid_test))

    # Factor Analsysis
    print("\n Performing factor analysis...\n")
    fa_train = FactorAnalyzer(n_factors=N_FACTORS, rotation="varimax")
    fa_train.fit(x_train)
    ev, v = fa_train.get_eigenvalues()

    print("Optimal Number Factors: {}".format(np.sum(ev>1)))
    
    # Scree plot of eigenvalues to show # factors for selection
    plt.scatter(range(1, x_train.shape[1]+1), ev)
    plt.plot(range(1, x_train.shape[1]+1), ev)
    plt.plot(np.sum(ev > 1), 1, 'ro')
    plt.plot(range(1, x_train.shape[1]+1), np.ones(x_train.shape[1]))
    plt.title('Scree Plot')
    plt.xlabel('Features')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

    # Get loadings and communalities
    loadings = fa_train.loadings_
    # plt.imshow(loadings)
    comm = fa_train.get_communalities()
    plt.hist(comm, bins=np.arange(min(comm), max(comm) + 0.01, 0.01))
    plt.xlabel('Sum of squared loadings (Common Variance)')
    plt.ylabel('Feature Count')
    plt.show()

    # Transform feature vectors based on model
    x_trainR = fa_train.transform(x_train).T
    x_testR = fa_train.transform(x_test).T

    # Save factor analysis model
    os.chdir('/home/nick/Documents/Hiwi/Factor Analysis/Models')
    filename = open(os.path.join(os.getcwd(), "FA-{}_Ehrlich2016.obj".format(N_FACTORS)), 'wb')
    pickle.dump(fa_train, filename)

    return x_trainR, y_train, x_testR, y_test

def main():
    x_trainR, y_train, x_testR, y_test = feature_extraction()
    os.chdir('/home/nick/Documents/Hiwi/Factor Analysis/Factors')
    savemat(os.path.join(os.getcwd(), "Ehrlich 2016/Training_Data.mat"), {"feats": x_trainR, "labels": y_train})
    savemat(os.path.join(os.getcwd(), "Ehrlich 2016/Test_Data.mat"), {"feats": x_testR, "labels": y_test})

if __name__ == "__main__":
    main()