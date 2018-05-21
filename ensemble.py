import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import models
from utils import RunningMean, use_gpu
from misc import FurnitureDataset, preprocess, preprocess_with_augmentation, NB_CLASSES, preprocess_hflip, preprocess_tencrop, preprocess_256crop, preprocess_288crop, preprocess_320crop, preprocess_352crop
import torch.utils.data as utils

from os import listdir
from os.path import isfile, join

BATCH_SIZE = 16

class EnsembleNet(nn.Module):

    def __init__(self, input_dim=5, output_dim=1):
        super(EnsembleNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear(x))
        return x


predictions = [f for f in listdir("train_predictions") if isfile(join("train_predictions", f))]
train_model_predictions = []
for file in predictions:
    train_model_predictions = torch.cat((train_model_predictions, torch.load(file)), dim=1)

train_pred = None
for train_pth in [f for f in listdir("train_predictions") if isfile(join("train_predictions", f))]:
    if train_pred is None:
        train_pred = torch.load(join('train_predictions',train_pth))
        train_pred = train_pred['px'].mean(dim=2).unsqueeze(dim=2)
        train_label = train_pred['lx']
    else:
        train_pred = torch.cat((train_pred,torch.load(join('train_predictions',train_pth))['px'].mean(dim=2).unsqueeze(dim=2)), dim=2)

val_pred = None
for val_pth in [f for f in listdir("val_predictions") if isfile(join("val_predictions", f))]
    if val_pred is None:
        val_pred = torch.load(join('val_predictions',val_pth))
        val_pred = val_pred['px'].mean(dim=2).unsqueeze(dim=2)
        val_pred = val_pred['lx']
    else:
        val_pred = torch.cat((val_pred,torch.load(join('val_predictions',val_pth))['px'].mean(dim=2).unsqueeze(dim=2)), dim=2)

train_dataset = utils.TensorDataset(train_pred['px'],train_pred['lx']) 
train_dataloader = utils.DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

val_dataset = utils.TensorDataset(val_pred['px'],val_pred['lx']) 
val_dataloader = utils.DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

model = EnsembleNet(5,1)
if use_gpu:
    model.cuda()

learning_rate = 1e-3
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

min_loss = float("inf")
for epoch in range(25):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for inputs, labels in pbar:

        inputs = Variable(inputs)
        labels = Variable(labels)
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = []
        for i in range(128):
            outputs.append(model(inputs))

        # test_prob = F.softmax(Variable(train_pred), dim=1).data.numpy()
        # train_predicted = np.argmax(test_prob, axis=1)
        # train_predicted += 1
        loss = criterion(Variable(torch.FloatTensor(outputs)), labels)
        running_loss.update(loss.item(), 1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description(f'{running_loss.value:.5f}')
    print(f'[+] epoch {epoch} {running_loss.value:.5f}')

    pbar = tqdm(val_dataloader, total=len(val_dataloader))
    for inputs, labels in pbar:

        inputs = Variable(inputs)
        labels = Variable(labels)
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = []
        for i in range(128):
            outputs.append(model(inputs))

    log_loss = criterion(Variable(torch.FloatTensor(outputs)), labels)
    log_loss = log_loss.data[0]
    _, preds = torch.max(px, dim=1)
    accuracy = torch.mean((preds != labels).float())
    print(f'[+] val {log_loss:.5f} {accuracy:.3f}')

    if log_loss < min_loss:
        torch.save(model.state_dict(), 'best_val_ensemble_weight.pth')
        torch.save(model.linear.weight, 'weighted_average.pth')
        print(f'[+] val score improved from {min_loss:.5f} to {log_loss:.5f}. Saved!')
        min_loss = log_loss
        patience = 0
    else:
        patience += 1
