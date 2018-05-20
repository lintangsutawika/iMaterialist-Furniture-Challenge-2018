import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.autograd import Variable

import models
import utils
from utils import RunningMean, use_gpu
from misc import FurnitureDataset, get_class_weights, preprocess, preprocess_with_augmentation, NB_CLASSES, preprocess_hflip, preprocess_tencrop, preprocess_256crop, preprocess_288crop, preprocess_320crop, preprocess_352crop

BATCH_SIZE = 16

def get_model():
    print('[+] loading model... ', end='', flush=True)
    # model = models.densenet201_finetune(NB_CLASSES)
    # model = models.dpn107_finetune(NB_CLASSES)
    # model = models.densenet161_finetune(NB_CLASSES)
    # model = models.se_resnext101_32x4d_finetune(NB_CLASSES)
    # model = models.nasnetalarge_finetune(NB_CLASSES)
    # model = models.inceptionresnetv2_finetune(NB_CLASSES)
    model = models.senet154_finetune(NB_CLASSES)
    if use_gpu:
        model.cuda()
    print('done')
    return model


def predict():
    model = get_model()
    model.load_state_dict(torch.load('best_val_weight.pth'))
    model.eval()

    tta_preprocess = [preprocess, preprocess_hflip, preprocess_tencrop, preprocess_256crop, preprocess_288crop, preprocess_320crop, preprocess_352crop]
    # tta_preprocess = [preprocess_tencrop, preprocess_256crop, preprocess_288crop, preprocess_320crop, preprocess_352crop]
    # tta_preprocess = [preprocess, preprocess_hflip, preprocess_tencrop, preprocess_320crop, preprocess_352crop]

    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = FurnitureDataset('test', transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=1,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)

    lx, px = utils.predict_tta(model, data_loaders)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    torch.save(data, 'test_prediction.pth')

    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = FurnitureDataset('val', transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=1,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)

    lx, px = utils.predict_tta(model, data_loaders)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    torch.save(data, 'val_prediction.pth')


def train():
    # train_dataset = FurnitureDataset('train', transform=preprocess_tencrop)
    train_dataset = FurnitureDataset('train', transform=preprocess_with_augmentation)
    val_dataset = FurnitureDataset('val', transform=preprocess_352crop)

    class_weight, dataset_length = get_class_weights('train')
    weighted_id = train_dataset.data['label_id'].apply(lambda x: class_weight[x-1])
    weighted_sampler = WeightedRandomSampler(weighted_id, dataset_length, replacement=True)
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=4,
                                      batch_size=BATCH_SIZE,
                                      sampler=weighted_sampler,
                                      shuffle=False)
                                      # shuffle=True)
    validation_data_loader = DataLoader(dataset=val_dataset, num_workers=1,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

    model = get_model()

    criterion = nn.CrossEntropyLoss().cuda()

    nb_learnable_params = sum(p.numel() for p in model.fresh_params())
    print(f'[+] nb learnable params {nb_learnable_params}')

    min_loss = float("inf")
    lr = 0
    patience = 0
    for epoch in range(25):
        print(f'epoch {epoch}')
        if epoch == 1:
            # lr = 0.00001
            # lr = 0.00003
            lr = 0.00003
            print(f'[+] set lr={lr}')
        if patience == 2:
            patience = 0
            model.load_state_dict(torch.load('best_val_weight.pth'))
            lr = lr / 10
            print(f'[+] set lr={lr}')
        if epoch == 0:
            lr = 0.001
            print(f'[+] set lr={lr}')
            optimizer = torch.optim.Adam(model.fresh_params(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

        running_loss = RunningMean()
        running_score = RunningMean()

        model.train()
        pbar = tqdm(training_data_loader, total=len(training_data_loader))
        for inputs, labels in pbar:
            batch_size = inputs.size(0)

            # #Compatible with TenCrops
            # try:
            #     bs, ncrops, c, h, w = inputs.size()
            # except:
            #     bs, c, h, w = inputs.size()

            # inputs = Variable(inputs)
            # labels = Variable(labels)
            # if use_gpu:
            #     inputs = inputs.cuda()
            #     labels = labels.cuda()

            # outputs = model(inputs.view(-1, c, h, w))

            # #Compatible with TenCrops
            # try:
            #     outputs = outputs.view(bs, ncrops, -1).mean(1) # avg over crops
            # except:
            #     outputs = outputs.view(bs,-1)

            inputs = Variable(inputs)
            labels = Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)

            loss = criterion(outputs, labels)
            running_loss.update(loss.item(), 1)
            running_score.update(torch.sum(preds != labels.data), batch_size)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f'{running_loss.value:.5f} {running_score.value:.3f}')
        print(f'[+] epoch {epoch} {running_loss.value:.5f} {running_score.value:.3f}')

        lx, px = utils.predict(model, validation_data_loader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.data[0]
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds != lx).float())
        print(f'[+] val {log_loss:.5f} {accuracy:.3f}')

        if log_loss < min_loss:
            torch.save(model.state_dict(), 'best_val_weight.pth')
            print(f'[+] val score improved from {min_loss:.5f} to {log_loss:.5f}. Saved!')
            min_loss = log_loss
            patience = 0
        else:
            patience += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    args = parser.parse_args()
    print(f'[+] start `{args.mode}`')
    if args.mode == 'train':
        train()
    elif args.mode == 'predict':
        predict()
