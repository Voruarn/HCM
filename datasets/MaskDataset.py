import matplotlib.pyplot as plt
import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import torch
import torchvision.transforms as transforms

class MaskDataset(torch.utils.data.Dataset):
    """一个用于加载 MaskDataset 的自定义数据集"""

    def __init__(self, data_path, img_size=512):
      
        self.features=[(data_path+i) for i in os.listdir(data_path) if i.split('.')[-1] in ['png','jpg','jpeg']]
        self.names=[i.split('/')[-1] for i in self.features]
        self.gt_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()])

        print('read ' + str(len(self.features)) + ' examples')

    def __getitem__(self, idx):
        gt=Image.open(self.features[idx]).convert('L')
        gt = self.gt_transform(gt)

        return {'mask':gt, 'name':self.names[idx]}
       
    def __len__(self):
        return len(self.features)
    










