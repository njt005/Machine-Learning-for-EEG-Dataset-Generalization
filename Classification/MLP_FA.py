#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 17, 2020

@author: Nick Tacca
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix

# Set initial variables
LR = 0.0001
WD = 0.004
EPOCHS = 100
BATCH_SIZE = 200
SEED = 4

class EEGdata(Dataset):
    """Loading EEG dataset"""

    def __init__(self, mat_file, transform=None):
        """
            mat_file (string): Path to the mat file with feature data and labels
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.data = loadmat(os.path.join(os.getcwd(), mat_file))
        self.feats = self.data["feats"]
        self.labels = self.data["labels"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.feats[:, idx]
        label = self.labels[idx]
        label = np.where(label == -1, 0, label)

        return sample, label

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, label):
        # swap color axis for torch files
        feat = sample.double()
        feat_tensor = torch.from_numpy(feat)
        label_tensor = torch.from_numpy(label)
        return feat_tensor, label_tensor

class MLP(nn.Module):
    """MLP for EEG classification"""

    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(111, 111),
            nn.ReLU(True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(111, 56),
            nn.ReLU(True)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(56, 2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.fc3(self.fc2(self.fc1(x)))
        return x
    
def train(model, epoch):
    train_loader = load_data("TRAIN")
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss()
    model.train()

    accuracy = []
    train_loss = []
    
    for batch, (x_train, y_train) in enumerate(train_loader):
        optimizer.zero_grad()
        output_train = model(x_train)
        y_train = y_train.long()
        loss_train = criterion(output_train, y_train.squeeze(1))
        loss_train.backward()
        train_loss.append(loss_train.item())
        optimizer.step()

        with torch.no_grad():
            output = model(x_train)
    
        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)

        # Accuracy on training set
        acc = accuracy_score(y_train, predictions)
        accuracy.append(acc)
    
    accuracy = np.vstack(accuracy)
    train_loss = np.vstack(train_loss)
    
    print('Epoch: {} Average loss: {:.2f}'.format(
        epoch,
        np.mean(train_loss)
    ))

    return train_loss, accuracy

def test(model):
    test_loader = load_data("TEST")
    
    true_neg = 0
    true_pos = 0

    for batch, (x_test, y_test) in enumerate(test_loader):

        with torch.no_grad():
            output = model(x_test)
        
        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)

        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

        true_neg += tn
        true_pos += tp
    
    true_negR = true_neg / (4*len(test_loader)) * 100
    true_posR = true_pos / (4*len(test_loader)) * 100

    return true_negR, true_posR

def device_settings(): 
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): 
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device

def load_data(dataset):

    os.chdir('/home/nick/Documents/Hiwi/Factor Analysis/Factors/Ehrlich 2016')

    if dataset == "TRAIN":
        # Load training data
        train_dataset = EEGdata(mat_file="Training_Data.mat", transform=transforms.ToTensor())
        dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    elif dataset == "TEST":
        # Load test data
        test_dataset = EEGdata(mat_file="Test_Data.mat", transform=transforms.ToTensor())
        dataset_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    return dataset_loader
    
def main():
    # device = device_settings()
    device = torch.device("cpu")
    mlp = MLP().to(device).double()

    loss = []
    acc = []

    # Training
    for epoch in range(1, EPOCHS + 1):
        train_loss, accuracy = train(mlp, epoch)
        loss.append(train_loss)
        acc.append(accuracy)
    
    accuracy_m = np.mean(accuracy)
    
    print("Final Training accuracy = {:.2f}%".format(accuracy_m*100))

    loss = np.vstack(loss)
    acc = np.vstack(acc)
    
    # Plotting results
    plt.plot(loss, label="Training Loss")
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(acc*100, label="Accuracy (%)")
    plt.ylim((0, 100))
    plt.legend()
    plt.grid()
    plt.show()

    # Test!
    true_negR, true_posR = test(mlp)

    # Get balanced accuracy
    accuracy_m = np.mean((true_negR, true_posR))
    print("\nTest Results")
    print("TNR: {:.2f}%, TPR: {:.2f}%, bAcc: {:.2f}%".format(true_negR, 
                                                            true_posR, 
                                                            accuracy_m))

    os.chdir('/home/nick/Documents/Hiwi/Models')
    torch.save(mlp.state_dict(), os.path.join(os.getcwd(), "MLP_FA_Ehrlich2016.pt"))
    
if __name__ == "__main__":
    main()
