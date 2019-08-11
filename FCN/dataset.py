# encoding: utf-8
"""
@author: FroyoZzz
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: froyorock@gmail.com
@software: garner
@file: dataset.py
@time: 2019-08-07 17:21
@desc:
"""
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, image_path = "data/BagImages", mode = "train"):
        assert mode in ("train", "val", "test")
        self.image_path = image_path
        self.image_list = glob(os.path.join(self.image_path, "*.jpg"))
        self.mode = mode

        if mode in ("train", "val"):
            self.mask_path = self.image_path + "Masks"

        self.transform_x = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([ T.ToTensor()])

    def __getitem__(self, index):
        if self.mode in ("train", "val"):
            image_name = self.image_list[index].split("\\")[-1].split(".")[0]
            X = Image.open(self.image_list[index])
            
            mask = np.array(Image.open(os.path.join(self.mask_path, image_name+".jpg")).convert('1').resize((256, 256)))
            masks = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=np.uint8)
            masks[:, :, 0] = mask
            masks[:, :, 1] = ~mask

            X = self.transform_x(X)
            masks = self.transform_mask(masks) * 255
            return X, masks
        
        else:
            X = Image.open(self.image_list[index])
            X = self.transform_x(X)
            path = self.image_list[index]
            return X, path

    def __len__(self):
        return len(self.image_list)
