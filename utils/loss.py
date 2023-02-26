import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class WeightedLoss(nn.Module):
    def __init__(self, weight = 0.5):
        super(WeightedLoss, self).__init__()
        self.weight = weight
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, inputs, targets):

        buff = nn.MaxPool2d(5, stride = 1, padding = 2)
        targets_buff = buff(targets.type(torch.float32))

        loss1 = self.criterion(inputs, targets)
        loss2 = self.criterion(inputs, targets_buff)

        loss = loss1 + self.weight * loss2

        return loss

class DeepCrackLoss(nn.Module):

    def __init__(self):
        super(DeepCrackLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, inputs, targets):
        n = inputs.shape[1]           # layers of outputs
        
        targets = targets.repeat([1,n,1,1])

        loss = self.criterion(inputs, targets)

        return loss
            


