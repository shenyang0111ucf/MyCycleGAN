from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


class GANDataset(Dataset):
    def __init__(self, root_type1, root_type2, transform=None):
        self.root_type1 = root_type1
        self.root_type2 = root_type2
        self.transform = transform

        self.type1_images = os.listdir(root_type1)
        self.type2_images = os.listdir(root_type2)
        self.type1_length = len(self.type1_images)
        self.type2_length = len(self.type2_images)
        self.length = max(len(self.type1_images), len(self.type2_images))

    def __len__(self):
        return self.length

    def readimage(self, root, images, images_length, index):
        img = images[index % images_length]
        img_path = os.path.join(root, img)
        img = np.array(Image.open(img_path).convert("RGB"))
        return img

    def __getitem__(self, index):
        type1_img = self.readimage(self.root_type1, self.type1_images, self.type1_length, index)
        type2_img = self.readimage(self.root_type2, self.type2_images, self.type2_length, index)
        if self.transform:
            augmentations = self.transform(image0=type1_img, image=type2_img)
            type1_img = augmentations["image0"]
            type2_img = augmentations["image"]

        return type1_img, type2_img
