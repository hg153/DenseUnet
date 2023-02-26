import torch
import numpy as np
from torch import nn

class weak_supervising_label():
    def __init__(self, labels):
        super(weak_supervising_label, self).__init__()
        self.labels = labels

        buff = nn.MaxPool2d(5, stride = 1, padding = 2)
        temp_buffer = buff(labels.type(torch.float32))
        self.buffer = temp_buffer - self.labels

    def update(self, pred, thres = 0.3):

        mask_background_pred = pred < thres

        self.buffer[mask_background_pred] = 0

        self.labels = self.labels + self.buffer



