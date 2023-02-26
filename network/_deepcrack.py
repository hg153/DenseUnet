import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel


class DeepCrackv1(_SimpleSegmentationModel):

    pass

class DeepCrackHead(nn.Module):

    def __init__(self):
        super(DeepCrackHead, self).__init__()
        # channels of each levels are [128, 256,512, 1024, 512]
        """
        ch = [128, 256,512, 1024, 512]
        self.m = nn.ModuleList(nn.Conv2d(x, 1, 1) for x in ch)  # output conv
        """
        self.detect = nn.Conv2d(5,1,1)

    def forward(self,x):
        """
        out = torch.Tensor().to(x[0].device)
        # features is a list of features from different levels
        for idx,_ in enumerate(x):
            out = torch.cat((out,self.m[idx](x[idx])), dim = 1) 
        """
        pred = self.detect(x)

        return torch.cat((x,pred), dim=1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



