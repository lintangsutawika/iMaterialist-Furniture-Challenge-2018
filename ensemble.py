import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import models
import utils
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
for train_pth in onlyfiles:
    if train_pred is None:
        train_pred = torch.load(join('train_predictions',train_pth))
        train_pred = train_pred['px'].mean(dim=2).unsqueeze(dim=2)
        train_label = train_pred['lx']
    else:
        train_pred = torch.cat((train_pred,torch.load(join('train_predictions',train_pth))['px'].mean(dim=2).unsqueeze(dim=2)), dim=2)


train_dataset = utils.TensorDataset(train_pred['px'],train_pred['lx']) 
train_dataloader = utils.DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False)
model = EnsembleNet(5,1)
if use_gpu:
    model.cuda()

learning_rate = 1e-3
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

for epoch in range(25):
    pbar = tqdm(training_data_loader, total=len(training_data_loader))
    for inputs, labels in pbar:

        inputs = Variable(inputs)
        labels = Variable(labels)
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        train = model(inputs)
        test_prob = F.softmax(Variable(train_pred), dim=1).data.numpy()
        train_predicted = np.argmax(test_prob, axis=1)
        train_predicted += 1
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(t, loss.item())
