import torch.nn as nn
import torch.nn.functional as F
from model.resnet import resnet50
from config.base_config import cfg
import torchvision.models as models
import torch

class EmbeddingNet(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        if backbone is None:
            backbone = models.efficientnet_b2(num_classes=1024, pretrained=True)
        # ct = 0
        # for child in backbone.features:
        #     ct += 1
        #     if ct < 8:
        #         for param in child.parameters():
        #             param.requires_grad = False
        
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.normalize(x, dim=1)
        return x



class EmbeddingResnet(nn.Module):
    def __init__(self):
        super(EmbeddingResnet, self).__init__()

        resnet = resnet50(pretrained=True)

        self.features = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1,
                                      resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool)
        # Fix blocks
        list_block = [0, 1, 4, 5, 6, 7]
        for block in list_block[:-3]:
            for p in self.features[block].parameters():
                p.requires_grad = False

        # for p in self.features[0].parameters():
        #     p.requires_grad = False
        # for p in self.features[1].parameters():
        #     p.requires_grad = False
        # if cfg.RESNET.FIXED_BLOCKS >= 3:
        #     for p in self.features[6].parameters():
        #         p.requires_grad = False
        # if cfg.RESNET.FIXED_BLOCKS >= 2:
        #     for p in self.features[5].parameters():
        #         p.requires_grad = False
        # if cfg.RESNET.FIXED_BLOCKS >= 1:
        #     for p in self.features[4].parameters():
        #         p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.features.apply(set_bn_fix)
        self.fc = nn.Linear(8192, 1024)

    def forward(self, x):
        features = self.features.forward(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = F.normalize(features, p=2, dim=1)
        return features


class EmbeddingLeNet(nn.Module):
    def __init__(self):
        super(EmbeddingLeNet, self).__init__()

        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 128),
                                nn.PReLU(),
                                nn.Linear(128, 64)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        output = F.normalize(output, p=2, dim=1)
        return output
