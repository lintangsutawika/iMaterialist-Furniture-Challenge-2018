import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

use_gpu = torch.cuda.is_available()


class RunningMean:
    def __init__(self, value=0, count=0):
        self.total_value = value
        self.count = count

    def update(self, value, count=1):
        self.total_value += value
        self.count += count

    @property
    def value(self):
        if self.count:
            return self.total_value / self.count
        else:
            return float("inf")

    def __str__(self):
        return str(self.value)


# def predict(model, dataloader):
#     all_labels = []
#     all_outputs = []
#     model.eval()
#
#     pbar = tqdm(dataloader, total=len(dataloader))
#     for inputs, labels in pbar:
#         all_labels.append(labels)
#
#         inputs = Variable(inputs, volatile=True)
#         if use_gpu:
#             inputs = inputs.cuda()
#
#         outputs = model(inputs)
#         all_outputs.append(outputs.data.cpu())
#
#     all_outputs = torch.cat(all_outputs)
#     all_labels = torch.cat(all_labels)
#     if use_gpu:
#         all_labels = all_labels.cuda()
#         all_outputs = all_outputs.cuda()
#
#     return all_labels, all_outputs

def predict(model, dataloader):
    all_labels = []
    all_outputs = []
    model.eval()

    pbar = tqdm(dataloader, total=len(dataloader))
    for inputs, labels in pbar:
        all_labels.append(labels)

        #Compatible with TenCrops
        try:
            bs, ncrops, c, h, w = inputs.size()
        except:
            bs, c, h, w = inputs.size()

        inputs = Variable(inputs)
        if use_gpu:
            inputs = inputs.cuda()

        # outputs = model(inputs)
        with torch.no_grad():
            outputs = model(inputs.view(-1, c, h, w))

        #Compatible with TenCrops
        try:
            outputs = outputs.view(bs, ncrops, -1).mean(1) # avg over crops
        except:
            outputs = outputs.view(bs,-1)

        all_outputs.append(outputs.data.cpu())
        # print(np.shape(result_avg.data[0]))

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    if use_gpu:
        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()

    return all_labels, all_outputs


def safe_stack_2array(a, b, dim=0):
    print("shape of a: {}".format(np.shape(a)))
    print("shape of b: {}".format(np.shape(b)))
    if a is None:
        return b.unsqueeze(-1)
    return torch.cat((a, b.unsqueeze(-1)), dim=dim)


def predict_tta(model, dataloaders):
    prediction = None
    lx = None
    for dataloader in dataloaders:
        lx, px = predict(model, dataloader)
        # prediction = safe_stack_2array(prediction, px)
        prediction = safe_stack_2array(prediction, px, dim=-1)

    return lx, prediction
