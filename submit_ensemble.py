#Averaged for each model, then averaged for all models before softmax
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
from misc import FurnitureDataset, preprocess

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('test_predictions') if isfile(join('test_predictions', f))]

# test_dataset = FurnitureDataset('test', transform=preprocess)
test_dataset = torch.load('test_dataset.pth')
# weighted_average = torch.load('weighted_average.pth')

test_pred = None
for test_pth in onlyfiles:
    if test_pred is None:
        test_pred = torch.load(join('test_predictions',test_pth))['px'].mean(dim=2).unsqueeze(dim=2)
    else:
        test_pred = torch.cat((test_pred,torch.load(join('test_predictions',test_pth))['px'].mean(dim=2).unsqueeze(dim=2)), dim=2)

test_pred = test_pred.mean(dim=2)
# test_pred = test_pred * weighted_average
test_prob = F.softmax(Variable(test_pred), dim=1).data.numpy()
test_predicted = np.argmax(test_prob, axis=1)
test_predicted += 1
result = test_predicted

sx = pd.read_csv('data/sample_submission_randomlabel.csv')
sx.loc[sx.id.isin(test_dataset.data.image_id), 'predicted'] = result
sx.to_csv('sx_ensemble.csv', index=False)
