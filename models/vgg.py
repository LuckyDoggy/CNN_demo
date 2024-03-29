import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]



class VGG(nn.Module):

    def __init__(self, data_channels,cfg, num_classes=10,batch_norm=False):
        super(VGG, self).__init__()
        self.features = self.make_layers(data_channels=data_channels,cfg=cfg,batch_norm=batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 , 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def make_layers(self,data_channels=3,cfg=None, batch_norm=False):
        layers = []
        in_channels=data_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(data_channels=3,cfg=cfg['A'],num_classes=10,batch_norm=False):
    model = VGG(data_channels=data_channels, cfg=cfg, num_classes=num_classes,batch_norm=batch_norm)
    return model

def vgg11_bn(data_channels=3,cfg=cfg['A'],num_classes=10,batch_norm=True):
    model = VGG(data_channels=data_channels, cfg=cfg, num_classes=num_classes,batch_norm=batch_norm)
    return model

def vgg13(data_channels=3,cfg=cfg['B'],num_classes=10,batch_norm=False):
    model = VGG(data_channels=data_channels, cfg=cfg, num_classes=num_classes,batch_norm=batch_norm)
    return model

def vgg13_bn(data_channels=3,cfg=cfg['B'],num_classes=10,batch_norm=True):
    model = VGG(data_channels=data_channels, cfg=cfg, num_classes=num_classes,batch_norm=batch_norm)
    return model

def vgg16(data_channels=3,cfg=cfg['D'],num_classes=10,batch_norm=False):
    model = VGG(data_channels=data_channels, cfg=cfg, num_classes=num_classes,batch_norm=batch_norm)
    return model

def vgg16_bn(data_channels=3,cfg=cfg['D'],num_classes=10,batch_norm=True):
    model = VGG(data_channels=data_channels, cfg=cfg, num_classes=num_classes,batch_norm=batch_norm)
    return model

def vgg19(data_channels=3,cfg=cfg['E'],num_classes=10,batch_norm=False):
    model = VGG(data_channels=data_channels, cfg=cfg, num_classes=num_classes,batch_norm=batch_norm)
    return model

def vgg19_bn(data_channels=3,cfg=cfg['E'],num_classes=10,batch_norm=True):
    model = VGG(data_channels=data_channels, cfg=cfg, num_classes=num_classes,batch_norm=batch_norm)
    return model

