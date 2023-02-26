import torch
from torch import nn
from torch.nn import functional as F
from .utils import _SimpleSegmentationModel


class FCN(_SimpleSegmentationModel):
    """
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass

class FCNHead(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(FCNHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,1, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class UNetHead(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(UNetHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels,1, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

