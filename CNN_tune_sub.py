#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 17:45:41 2024

Convolution neural network for digit recognition on mnist dataset

@author: ganchenko
"""
import torch

from tqdm import tqdm
from time import time
import numpy as np
import pandas as pd
import seaborn as sns
import random
from matplotlib import pyplot as plt 

import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()
from torchvision.transforms import v2

from sklearn.metrics import confusion_matrix
from matplotlib.colors import LogNorm
from argparse import ArgumentParser

def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    return np.mean(np.equal(predictions.numpy(), y.numpy()))

def plot_digits(X, img_pix=28, n=1, labels_predict=np.array([None]), labels_act=np.array([None]), rnd=True):
    """
    Plot n*n random digits from set X
    
    Args:
        X -  (num, img_pix*img_pix), array of digits
        img_pix - int, size of the digit img_pix*img_pix
        n - int, n is a number of columns and rows
        labels_predicted - (num, ) of int or str with predicted labels
        labels_act - (num, ) of int or str with actual labels
        rnd - Boolean, random chosing of samples
    Returns:
        nothing
    """
    plt.figure(figsize=[1.5*n, 1.5*n])
   
    for i in range(n*n):
        plt.subplot(n , n, i+1)
        if rnd:
            indx=random.randint(0, len(X)-1)
        else:
            indx=i
        dig=X[indx].reshape(img_pix, img_pix)
        plt.imshow(dig, cmap='gray_r')
        plt.axis('off')
        str_title=''
        if labels_act[0]!=None: str_title+=' Act_num: '+str(labels_act[indx])+', \n'
        if labels_predict[0]!=None: str_title+='Pre_num: '+str(labels_predict[indx])
        plt.title(str_title)
            
    plt.tight_layout()        
    plt.show()       
    
class Flatten(nn.Module):
    """A custom layer that views an input as 1D."""
    def forward(self, input):
        return input.view(input.size(0), -1)

class ImageDataset(Dataset):

    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx] 
        image = torch.from_numpy(image).float()
        image = image.view(1, 28, 28)
        
        if self.transform is not None:
            image=self.transform(image)
    
        if self.labels is not None:  
            label = torch.tensor(self.labels[idx])
            return image, label
        return image

# Training Procedure
def train_model(train_data, dev_data, model, device=torch.device('cpu'), lr=0.01, momentum=0.9, nesterov=False, n_epochs=20, batch_size=32):
    """Train a model for N epochs given data and hyper-params."""

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
    train_accuracies = []
    val_accuracies =[]
    for epoch in range(1, n_epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Run **training***
        loss, acc = run_epoch(train_data, model.train(), optimizer, device, batch_size)
        print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(loss, acc))
        train_accuracies.append(acc)

        # Run **validation**
        val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer, device, batch_size)
        print('Val loss:   {:.6f} | Val accuracy:   {:.6f}'.format(val_loss, val_acc))
        val_accuracies.append(val_acc)
        # Save model
        torch.save(model, 'mnist_cnn.pt')
    return train_accuracies, val_accuracies

def run_epoch(data, model, optimizer, device=torch.device('cpu'), batch_size=32):
    """Train model for one pass of train data, and return loss, acccuracy"""
   
    losses = []
    batch_accuracies = []
    is_training = model.training
    
    # batchify data
    #indices permutation in training set for a better SGD estimation
    if is_training: 
        data_batched = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        data_batched = DataLoader(data, batch_size=batch_size)
    
    # Iterate through batches
    for batch in tqdm(data_batched):
        
        x, y = batch
        # Move data to the specified device (GPU or CPU)
        x, y = x.to(device), y.to(device)
        out = model(x)

        predictions = torch.argmax(out, dim=1)
        batch_accuracies.append(compute_accuracy(predictions.cpu(), y.cpu()))

        loss = F.cross_entropy(out, y)
        losses.append(loss.data.item())

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(batch_accuracies)
    return avg_loss, avg_accuracy

def main(args):
    device=torch.device(args.device)
    n_epochs=args.epochs
    batch_size=args.batch_size

    #import data
    all_data = pd.read_csv("train.csv")
    y=all_data.label.to_numpy(dtype='int64')
    img_pix=28

    X=all_data.drop(['label'], axis=1, inplace=False).to_numpy(dtype='float')
    # Scaling data to [0, 1]
    X=X/np.max(X)

    # Set transform to augment data for training 
    transf = v2.RandomAffine(degrees=20, scale=(0.9, 1.1))
    X_set = ImageDataset(X, y, transform=transf)

    tmp_images = [X_set[i][0] for i in range(9)]
    tmp_labels = [X_set[i][1].numpy() for i in range(9)]
#    plot_digits(tmp_images, n=3, labels_act=tmp_labels, rnd=False) uncomment for debuging

    X_train_set, X_dev_set=random_split(X_set, [0.9, 0.1], generator=torch.Generator().manual_seed(42))


    #%% Train model

    #################################
    ## Model specification
    model = nn.Sequential(
	      nn.Conv2d(1, 32, (3, 3)),
	      nn.ReLU(),
	      nn.MaxPool2d((2, 2)),
	      nn.Conv2d(32, 64, (3, 3)),
	      nn.ReLU(),
	      nn.MaxPool2d((2, 2)),
	      Flatten(),
	      nn.Linear(64*5*5,128),
	      nn.Dropout(),
	      nn.Linear(128, 10),
	    )
    ##################################

#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print('Program will run on: ', device)
    model = model.to(device)

#    batch_size = 128
#    n_epochs = 40
    start_time=time()
    train_acc, dev_acc=train_model(X_train_set, X_dev_set, 
				   model, device=device, 
				   nesterov=True, 
				   n_epochs=n_epochs, batch_size=batch_size)
    print(f"Model took {(time() - start_time):.2f} seconds to do {n_epochs} epochs with batch size {args.batch_size} on {args.device}.")
    #%% Plot training and validation accuracies in epochs #uncomment for debuging
#    plt.subplot()
#    plt.plot(np.arange(1, n_epochs+1), train_acc, 'b', label='Train accuracy')
#    plt.plot(np.arange(1, n_epochs+1), dev_acc, 'r', label='Validation accuracy')
#    plt.legend()
#    plt.xlabel('Epochs')
#    plt.ylabel('Accuracy')
#    plt.show()

    #%% Analysis of prediction results on dev data
    indxs=np.arange(0,len(X))
    random.shuffle(indxs)
    out=model(torch.tensor(X[indxs[:10000]].reshape(-1,1,img_pix,img_pix), dtype=torch.float).to(device))

    predictions = torch.argmax(out.cpu(), dim=1).numpy()
    confusion_mtx = confusion_matrix(predictions, y[indxs[:10000]])

    f,ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,  fmt= '1', ax=ax, norm=LogNorm(), cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    #%% Evaluate the model on test data and save submission.cvs

#    test_data = pd.read_csv("test.csv")
#    X_test=test_data.to_numpy(dtype='float')

    #Train model on the validation dataset
#    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

#    for epoch in range(1, n_epochs+1):
#        print("-------------\nEpoch {}:\n".format(epoch))
#        loss, acc = run_epoch(X_dev_set, model.train(), optimizer, device, batch_size)
#        print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(loss, acc))

    # Scaling test data to [0, 1]
#    X_test=X_test/np.max(X_test)
#    X_test=X_test.reshape(-1, 1, img_pix, img_pix)

#    model_eval=model.eval()

#    out=model_eval(torch.tensor(X_test, dtype=torch.float).to(device))
	
#    predict = torch.argmax(out.cpu(), dim=1).numpy()

#    predictions = [int(x) for x in predict]

#    output = pd.DataFrame({'ImageId': range(1, len(X_test)+1), 'Label': predictions})
#    output.to_csv('submission_digits_CNN4.csv', index=False)
#    print("Your submission was successfully saved!")

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, required=True, choices=('cpu', 'cuda', 'mps'), help="Which device to use.")
    parser.add_argument('--batch_size', type=int, default=32, required=False, help="Training or inference batch size")
    parser.add_argument('--epochs', type=int, default=40, required=False, help="Number of step to train.")
    args = parser.parse_args()
    main(args)
