import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
from misc import FurnitureDataset, preprocess

test_dataset = FurnitureDataset('val', transform=preprocess)

test_pred = torch.load('val_prediction.pth')
test_prob = F.softmax(Variable(test_pred['px']), dim=1).data.numpy()
test_prob = test_prob.mean(axis=2)

test_predicted = np.argmax(test_prob, axis=1)
test_predicted += 1
result = test_predicted

sx = pd.read_csv('validation.csv')
sx.loc[sx.id.isin(test_dataset.data.image_id), 'predicted'] = result
sx.to_csv('val.csv', index=False)
