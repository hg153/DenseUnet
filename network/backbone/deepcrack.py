from torch import nn
import torch.nn.functional as F
import torch

class DeepCrack(nn.Module):
    def __init__(self, num_classes = 2):
        super(DeepCrack, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.deconv1 = nn.ConvTranspose2d(128, 1, 3, stride = 1,padding =1)
        self.deconv2 = nn.ConvTranspose2d(256, 1, 3, stride = 1,padding =1)
        self.deconv3 = nn.ConvTranspose2d(512, 1, 3, stride = 1,padding =1)
        self.deconv4 = nn.ConvTranspose2d(1024, 1, 3, stride = 1,padding =1)
        self.deconv5 = nn.ConvTranspose2d(512, 1, 3, stride = 1,padding =1)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.layer1(x)
        l1 = out
        out = self.pool(out)

        out = self.layer2(out)
        l2 = out
        out = self.pool(out)

        out = self.layer3(out)
        l3 = out
        out = self.pool(out)

        out = self.layer4(out)
        l4 = out
        out = self.pool(out)

        out = self.layer5(out)
        l5 = out

        out = F.interpolate(out, size = l4.shape[2:], mode = 'bilinear', align_corners = False)
        out = self.layer6(out)
        l6 = out

        out = F.interpolate(out, size = l3.shape[2:], mode = 'bilinear', align_corners = False)
        out = self.layer7(out)
        l7 = out

        out = F.interpolate(out, size = l2.shape[2:], mode = 'bilinear', align_corners = False)
        out = self.layer8(out)
        l8 = out

        out = F.interpolate(out, size = l1.shape[2:], mode = 'bilinear', align_corners = False)
        out = self.layer9(out)
        l9 = out

        """
        l2 = F.interpolate(l2, size = l1.shape[2:], mode = 'bilinear', align_corners = False)
        l3 = F.interpolate(l3, size = l1.shape[2:], mode = 'bilinear', align_corners = False)
        l4 = F.interpolate(l4, size = l1.shape[2:], mode = 'bilinear', align_corners = False)
        l5 = F.interpolate(l5, size = l1.shape[2:], mode = 'bilinear', align_corners = False)
        l6 = F.interpolate(l6, size = l1.shape[2:], mode = 'bilinear', align_corners = False)
        l7 = F.interpolate(l7, size = l1.shape[2:], mode = 'bilinear', align_corners = False)
        l8 = F.interpolate(l8, size = l1.shape[2:], mode = 'bilinear', align_corners = False)
        """

        f1 = torch.cat([l1,l9], dim = 1)
        f2 = torch.cat([l2,l8], dim = 1)
        f3 = torch.cat([l3,l7], dim = 1)
        f4 = torch.cat([l4,l6], dim = 1)
        f5 = l5

        f2 = F.interpolate(f2, size = l1.shape[2:], mode = 'bilinear', align_corners = False)
        f3 = F.interpolate(f3, size = l1.shape[2:], mode = 'bilinear', align_corners = False)
        f4 = F.interpolate(f4, size = l1.shape[2:], mode = 'bilinear', align_corners = False)
        f5 = F.interpolate(f5, size = l1.shape[2:], mode = 'bilinear', align_corners = False)

        f1 = self.deconv1(f1)
        f2 = self.deconv2(f2)
        f3 = self.deconv3(f3)
        f4 = self.deconv4(f4)
        f5 = self.deconv5(f5)

        out = torch.cat([f1,f2,f3,f4,f5], dim = 1)

        return out

def deepcrack():

    model = DeepCrack()

    return model