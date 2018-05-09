"""
@author Konstantin Lopuhin
"""

from functools import partial

from torch import nn
import torchvision.models as M
import pretrainedmodels

resnet18 = M.resnet18
resnet34 = M.resnet34
resnet50 = M.resnet50
resnet101 = M.resnet101
resnet152 = M.resnet152
vgg16 = M.vgg16
vgg16_bn = M.vgg16_bn
densenet121 = M.densenet121
densenet161 = M.densenet161
densenet201 = M.densenet201


class ResNetFinetune(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = net_cls(pretrained=True)
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


class DenseNetFinetune(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.densenet121):
        super().__init__()
        self.net = net_cls(pretrained=True)
        self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        return self.net(x)


class InceptionV3Finetune(nn.Module):
    finetune = True

    def __init__(self, num_classes: int):
        super().__init__()
        self.net = M.inception_v3(pretrained=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        if self.net.training:
            x, _aux_logits = self.net(x)
            return x
        else:
            return self.net(x)


class FinetunePretrainedmodels(nn.Module):
    finetune = True

    def __init__(self, num_classes: int, net_cls, net_kwards):
        super().__init__()
        self.net = net_cls(**net_kwards)
        self.net.last_linear = nn.Linear(self.net.last_linear.in_features, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class DualPathNetworkFinetune(nn.Module):
    finetune = True

    def __init__(self, num_classes: int, net_cls, net_kwards):
        super().__init__()
        self.net = net_cls(**net_kwards)
        self.net.classifier = nn.Conv2d(self.net.classifier.in_channels, num_classes, self.net.classifier.kernel_size, self.net.classifier.stride)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        return self.net(x)


resnet18_finetune = partial(ResNetFinetune, net_cls=M.resnet18)
resnet34_finetune = partial(ResNetFinetune, net_cls=M.resnet34)
resnet50_finetune = partial(ResNetFinetune, net_cls=M.resnet50)
resnet101_finetune = partial(ResNetFinetune, net_cls=M.resnet101)
resnet152_finetune = partial(ResNetFinetune, net_cls=M.resnet152)

densenet121_finetune = partial(DenseNetFinetune, net_cls=M.densenet121)

densenet161_finetune = partial(DenseNetFinetune, net_cls=M.densenet161)
densenet201_finetune = partial(DenseNetFinetune, net_cls=M.densenet201)

xception_finetune = partial(FinetunePretrainedmodels,
                            net_cls=pretrainedmodels.xception,
                            net_kwards={'pretrained': 'imagenet'})

inceptionresnetv2_finetune = partial(FinetunePretrainedmodels,
                            net_cls=pretrainedmodels.inceptionresnetv2,
                            net_kwards={'pretrained': 'imagenet'})

nasnetalarge_finetune = partial(FinetunePretrainedmodels,
                            net_cls=pretrainedmodels.nasnetalarge,
                            net_kwards={'num_classes':1000, 'pretrained': 'imagenet'})

se_resnext101_32x4d_finetune = partial(FinetunePretrainedmodels,
                            net_cls=pretrainedmodels.se_resnext101_32x4d,
                            net_kwards={'num_classes':1000, 'pretrained': 'imagenet'})

dpn107_finetune = partial(DualPathNetworkFinetune,
                            net_cls=pretrainedmodels.dpn107,
                            net_kwards={'pretrained': 'imagenet+5k'})
