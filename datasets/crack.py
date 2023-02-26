import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from torch import nn
from tqdm import tqdm


class Crack(data.Dataset):
    """
    Crack dataset
    """
    CrackClass = namedtuple('CrackClass', ['name', 'id', 'train_id','color' ])
    classes = [CrackClass('background', 0, 0, (0,0,0)),
               CrackClass('crack', 1, 1, (150,150,150))
               ]

    train_id_to_color = [c.color for c in classes]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gt'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'images', split)

        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.buff = nn.MaxPool2d(5, stride = 1, padding = 2)

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        

        for file_name in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, file_name))
            self.targets.append(os.path.join(self.targets_dir, file_name[:-4]+ '.png'))

        self.gts = []
        self.buffer = []
        for item in self.targets:
            gt = torch.from_numpy( np.array( Image.open(item), dtype='uint8') )
            self.gts.append(gt)

            temp_buffer = self.buff(gt.type(torch.float32).unsqueeze(0)).squeeze(0)
            self.buffer.append((temp_buffer - gt).type(torch.bool))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = self.gts[index]
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)

    def update_gt(self, model, transform = None, thres = 0.3, device = None):
        model.eval()
        for idx, img_path in tqdm(enumerate(self.images)):
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)

            pred = model(img).sigmoid().detach().max(1)[0].cpu().squeeze(0) # HW
            pred[pred>=thres] = 1
            pred[pred<thres] = 0

            self.gts[idx][self.buffer[idx]] = pred[self.buffer[idx]].type(torch.uint8)






