import json
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from augmentation import HorizontalFlip
from sklearn.utils import class_weight

NB_CLASSES = 128
IMAGE_SIZE = 224 #SeNet, se_resnext101
# IMAGE_SIZE = 299
# IMAGE_SIZE = 331 #NasNet
# IMAGE_SIZE = 320

class FurnitureDataset(Dataset):
    def __init__(self, preffix: str, transform=None):
        self.preffix = preffix
        if preffix == 'val':
            path = 'validation'
        else:
            path = preffix
        path = f'data/{path}.json'
        self.transform = transform
        img_idx = {int(p.name.split('.')[0])
                   for p in Path(f'data/{preffix}').glob('*.jpg')}
        data = json.load(open(path))
        if 'annotations' in data:
            data = pd.DataFrame(data['annotations'])
        else:
            data = pd.DataFrame(data['images'])
        self.full_data = data
        nb_total = data.shape[0]
        data = data[data.image_id.isin(img_idx)].copy()
        data['path'] = data.image_id.map(lambda i: f"data/{preffix}/{i}.jpg")
        self.data = data
        print(f'[+] dataset `{preffix}` loaded {data.shape[0]} images from {nb_total}')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row['path'])
        if self.transform:
            img = self.transform(img)

        target = row['label_id'] - 1 if 'label_id' in row else -1
        return img, target
        
def get_class_weights(preffix):

    # data = pd.read_json(open(preffix,'r'))
    # y_train = []
    # for entry in data['annotations']:
    #     y_train.append(entry['label_id'])
    # y_train = np.asarray(y_train)

    path = preffix
    path = f'data/{path}.json'
    img_idx = {int(p.name.split('.')[0])
               for p in Path(f'data/{preffix}').glob('*.jpg')}
    data = json.load(open(path))
    if 'annotations' in data:
        data = pd.DataFrame(data['annotations'])
    data = data[data.image_id.isin(img_idx)].copy()

    y_train = data['label_id']

    return class_weight.compute_class_weight('balanced', np.unique(y_train), y_train), len(y_train)


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize
])
preprocess_hflip = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    HorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
preprocess_with_augmentation = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3,
                           contrast=0.3,
                           saturation=0.3),
    transforms.ToTensor(),
    normalize
])

preprocess_tencrop = transforms.Compose([
    transforms.Resize((IMAGE_SIZE*2, IMAGE_SIZE*2)),
    transforms.TenCrop(IMAGE_SIZE),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
])

preprocess_256crop = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.TenCrop(IMAGE_SIZE),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
])

preprocess_288crop = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.TenCrop(IMAGE_SIZE),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
])

preprocess_320crop = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.TenCrop(IMAGE_SIZE),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
])

preprocess_352crop = transforms.Compose([
    transforms.Resize((352, 352)),
    transforms.TenCrop(IMAGE_SIZE),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
])
