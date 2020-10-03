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

N_FACTORS = 7

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
    x_train = x_train[:, 19, :].T
    x_test = x_test[:, 19, :].T

    x_train_m = np.mean(x_train, axis=0, keepdims=True)
    x_test_m = np.mean(x_test, axis=0, keepdims=True)

    # Visualize initial feature vectors
    fig, ax = plt.subplots()
    im = plt.imshow(x_train_m)
    plt.title('Initial Mean Training Feature Vector (P3)')
    plt.ylabel('P3')
    plt.xlabel('All Channel Features')
    plt.yticks([])
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()

    fig, ax = plt.subplots()
    plt.title('Initial Mean Testing Feature Vector (P3)')
    im = plt.imshow(x_test_m)
    plt.ylabel('P3')
    plt.xlabel('All Channel Features')
    plt.yticks([])
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()

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

    # Visualize loadings
    fig, ax = plt.subplots()
    im = plt.imshow(loadings.T)
    plt.title('Factor Loadings')
    plt.ylabel('Latent Factors')
    plt.xlabel('Input Data')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()

    comm = fa_train.get_communalities()
    plt.hist(comm, bins=np.arange(min(comm), max(comm) + 0.01, 0.01))
    plt.xlabel('Sum of squared loadings (Common Variance)')
    plt.ylabel('Feature Count')
    plt.show()

    # Transform feature vectors based on model
    x_trainR = fa_train.transform(x_train).T
    x_testR = fa_train.transform(x_test).T

    x_trainR_m = np.mean(x_trainR, axis=1, keepdims=True)
    x_testR_m = np.mean(x_testR, axis=1, keepdims=True)

    # Visualize transformed feature vectors
    fig, ax = plt.subplots()
    im = plt.imshow(x_trainR_m.T)
    plt.title('Transformed Mean Training Feature Vector (P3)')
    plt.ylabel('P3')
    plt.xlabel('All Channel Features')
    plt.yticks([])
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()

    fig, ax = plt.subplots()
    im = plt.imshow(x_testR_m.T)
    plt.title('Transformed Mean Testing Feature Vector (P3)')
    plt.ylabel('P3')
    plt.xlabel('All Channel Features')
    plt.yticks([])
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()

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