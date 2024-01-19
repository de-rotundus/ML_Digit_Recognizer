#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 17:45:41 2024

Convolution neural network for digit recognition on mnist dataset

@author: ganchenko
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import random
from matplotlib import pyplot as plt 
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, Subset

from sklearn.metrics import confusion_matrix
from matplotlib.colors import LogNorm

def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    return np.mean(np.equal(predictions.numpy(), y.numpy()))

def plot_digits(X, img_pix, n=1, labels_predict=np.array([None]), labels_act=np.array([None]), rnd=True):
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
            indx=random.randint(0, X.shape[0]-1)
        else:
            indx=i
        dig=X[indx, :].reshape(img_pix, img_pix)
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
        loss, acc = run_epoch(train_data, model.train(), optimizer, device)
        print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(loss, acc))
        train_accuracies.append(acc)

        # Run **validation**
        val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer, device)
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
        batch_accuracies.append(compute_accuracy(predictions, y))

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

#import data
all_data = pd.read_csv("train.csv")
y=all_data.label.to_numpy(dtype='int64')

img_pix=28


X=all_data.drop(['label'], axis=1, inplace=False).to_numpy(dtype='float')
# Scaling data to [0, 1]
X=X/np.max(X)
#plot_digits(X, img_pix, 3, labels_act=y)

X_set = ImageDataset(X, y)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

batch_size = 32
n_epochs = 20
train_acc, dev_acc=train_model(X_train_set, X_dev_set, 
                               model, device=device, 
                               nesterov=True, 
                               n_epochs=n_epochs, batch_size=batch_size)

#%% Plot training and validation accuracies in epochs
plt.subplot()
plt.plot(np.arange(1, n_epochs+1), train_acc, 'b', label='Train accuracy')
plt.plot(np.arange(1, n_epochs+1), dev_acc, 'r', label='Validation accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

#%% Analysis of prediction results on dev data
indxs=np.arange(0,len(X))
random.shuffle(indxs)
out=model(torch.tensor(X[indxs[:10000]].reshape(-1,1,img_pix,img_pix), dtype=torch.float32))

predictions = torch.argmax(out, dim=1).numpy()
confusion_mtx = confusion_matrix(predictions, y[indxs[:10000]])

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,  fmt= '1', ax=ax, norm=LogNorm(), cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

#%% Evaluate the model on test data and save submission.cvs

test_data = pd.read_csv("test.csv")

X_test=test_data.to_numpy(dtype='float')

# Scaling data to [0, 1]
X_test=X_test/np.max(X_test)
X_test=X_test.reshape(-1, 1, img_pix, img_pix)

model_eval=model.eval()

out=model_eval(torch.tensor(X_test, dtype=torch.float32))
    
predict = torch.argmax(out, dim=1).numpy()

predictions = [int(x) for x in predict]

output = pd.DataFrame({'ImageId': range(1, len(X_test)+1), 'Label': predictions})
output.to_csv('submission_digits_CNN4.csv', index=False)
print("Your submission was successfully saved!")
