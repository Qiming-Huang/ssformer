import os
import numpy as np
from numpy.random.mtrand import seed
from torch.utils.data import Dataset
from PIL import Image
import torch
import yaml
import sys
import albumentations as A
import torchvision.transforms as transforms
from utils.test_transforms import *


# final version of dataset
import cv2        
class CustomDataSet(Dataset):
    def __init__(self, img_path, label_path, transform_list):
        self.img_path = img_path
        self.label_path = label_path
        self.img_files = self.read_file(self.img_path)
        self.label_files = self.read_file(self.label_path)
        self.transform = self.get_transform(transform_list)
        self.image_size = []
        self.image_name = []

        # 初始化图片大小
        self._init_img_size()
        

    def __getitem__(self, index):

        img = Image.open(self.img_files[index]).convert('RGB')
        label = Image.open(self.label_files[index]).convert('L')
        name = self.img_files[index].split('/')[-1]
#        self.image_name.append(name)
        
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'          
        shape = label.size[::-1]
        sample = {'image': img, 'gt': label, 'name': name, 'shape': shape}            
        sample = self.transform(sample)
        img, label = sample['image'],sample['gt']
        img = torch.as_tensor(img)
        label = torch.as_tensor(label)   
        return img.float(), label.float()

    def __len__(self):
        total_img = len(self.img_files)
        return total_img


    def _init_img_size(self):
        for i in range(self.__len__()):
            img = Image.open(self.img_files[i])
            self.image_size.append(img.size)
            name = self.img_files[i].split('/')[-1]
            self.image_name.append(name)


    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list
        
    @staticmethod
    def get_transform(transform_list):
        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)
