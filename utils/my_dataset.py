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
from utils.custom_transforms import *

#f = open(sys.argv[1])
#config = yaml.safe_load(f)

# Datareaderset for isic2018
class ISIC2018(Dataset):
    def __init__(self, train_img_root, val_img_root, train_label_root, val_label_root, crop_size, mode='train'):
        self.train_img_files = self.read_file(train_img_root)
        self.val_img_files = self.read_file(val_img_root)
        self.train_label_files = self.read_file(train_label_root)
        self.val_label_files = self.read_file(val_label_root)
        self.mode = mode
        self.crop_size = crop_size

    def __getitem__(self, index):
        if self.mode == 'train':
            img = Image.open(self.train_img_files[index])
            label = Image.open(self.train_label_files[index])

            img = img.resize((self.crop_size[0], self.crop_size[1]))
            label = label.resize((self.crop_size[0], self.crop_size[1]))

            img = np.array(img) / 255
            label = np.array(label)

            img = torch.as_tensor(img)
            label = torch.as_tensor(label)

            img = img.permute(2, 0, 1)

            return img.float(), label.long()

        if self.mode == 'val':
            img = Image.open(self.val_img_files[index])
            label = Image.open(self.val_label_files[index])

            img = img.resize((self.crop_size[0], self.crop_size[1]))
            label = label.resize((self.crop_size[0], self.crop_size[1]))

            img = np.array(img) / 255
            label = np.array(label)

            img = torch.as_tensor(img)
            label = torch.as_tensor(label)

            print(img.shape)
            img = img.permute(2, 0, 1)

            return img.float(), label.long()

    def __len__(self):
        if self.mode == 'train':
            total_img = len(self.train_img_files)
            return total_img
        if self.mode == 'val':
            total_img = len(self.val_img_files)
            return total_img

    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

# dataset fro Kvasir
class Kvasir(Dataset):
    def __init__(self, train_img_root, val_img_root, train_label_root, val_label_root, crop_size, mode='train'):
        self.train_img_files = self.read_file(train_img_root)
        self.val_img_files = self.read_file(val_img_root)
        self.train_label_files = self.read_file(train_label_root)
        self.val_label_files = self.read_file(val_label_root)
        self.mode = mode
        self.crop_size = crop_size

    def __getitem__(self, index):
        if self.mode == 'train':
            img = Image.open(self.train_img_files[index])
            label = Image.open(self.train_label_files[index])

            img = img.resize((self.crop_size[0], self.crop_size[1]))
            label = label.resize((self.crop_size[0], self.crop_size[1]))

            img = np.array(img) / 255
            label = np.array(label)

            print(np.max(label))

            img = torch.as_tensor(img)
            label = torch.as_tensor(label)

            img = img.permute(2, 0, 1)

            return img.float(), label.long()

        if self.mode == 'val':
            img = Image.open(self.val_img_files[index])
            label = Image.open(self.val_label_files[index])

            img = img.resize((self.crop_size[0], self.crop_size[1]))
            label = label.resize((self.crop_size[0], self.crop_size[1]))

            img = np.array(img) / 255
            label = np.array(label)

            img = torch.as_tensor(img)
            label = torch.as_tensor(label)

            img = img.permute(2, 0, 1)

            return img.float(), label.long()

    def __len__(self):
        if self.mode == 'train':
            total_img = len(self.train_img_files)
            return total_img
        if self.mode == 'val':
            total_img = len(self.val_img_files)
            return total_img

    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

# dataset fro CVC-ClinicDB
class CVC(Dataset):
    def __init__(self, train_img_root, val_img_root, train_label_root, val_label_root, crop_size, mode='train'):
        self.train_img_files = self.read_file(train_img_root)
        self.val_img_files = self.read_file(val_img_root)
        self.train_label_files = self.read_file(train_label_root)
        self.val_label_files = self.read_file(val_label_root)
        self.mode = mode
        self.crop_size = crop_size

    def __getitem__(self, index):
        if self.mode == 'train':
            img = Image.open(self.train_img_files[index])
            label = Image.open(self.train_label_files[index])

            img = img.resize((self.crop_size[0], self.crop_size[1]))
            label = label.resize((self.crop_size[0], self.crop_size[1]))

            img = np.array(img) / 255
            label = np.array(label)

            img = torch.as_tensor(img)
            label = torch.as_tensor(label)

            img = img.permute(2, 0, 1)

            return img.float(), label.long()

        if self.mode == 'val':
            img = Image.open(self.val_img_files[index])
            label = Image.open(self.val_label_files[index])

            img = img.resize((self.crop_size[0], self.crop_size[1]))
            label = label.resize((self.crop_size[0], self.crop_size[1]))

            img = np.array(img) / 255
            label = np.array(label)

            img = torch.as_tensor(img)
            label = torch.as_tensor(label)

            img = img.permute(2, 0, 1)

            return img.float(), label.long()

    def __len__(self):
        if self.mode == 'train':
            total_img = len(self.train_img_files)
            return total_img
        if self.mode == 'val':
            total_img = len(self.val_img_files)
            return total_img

    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list


# build dataset
class DataBuilder(Dataset):
    def __init__(self, train_img_root, val_img_root, train_label_root, val_label_root, crop_size, mode='train'):
        self.train_img_files = self.read_file(train_img_root)
        self.val_img_files = self.read_file(val_img_root)
        self.train_label_files = self.read_file(train_label_root)
        self.val_label_files = self.read_file(val_label_root)
        self.mode = mode
        self.crop_size = crop_size

    def __getitem__(self, index):
        if self.mode == 'train':
            img = Image.open(self.train_img_files[index])
            label = Image.open(self.train_label_files[index])

            img = img.resize((self.crop_size[0], self.crop_size[1]))
            label = label.resize((self.crop_size[0], self.crop_size[1]))

            img = np.array(img) / 255
            label = np.array(label)

            if 'cvc' in config['dataset']['train_img_root']:
                # just for cvc start
                label = label[:,:,0]
                # just for cvc end
            img = torch.as_tensor(img)
            label = torch.as_tensor(label)
            if 'Seg' in config['dataset']['train_img_root'] or 'BRATS2015' in config['dataset']['train_img_root']:
                img = img.unsqueeze(0)
            else:
                img = img.permute(2, 0, 1)

            return img.float(), label.long()

        if self.mode == 'val':
            img = Image.open(self.val_img_files[index])
            label = Image.open(self.val_label_files[index])

            img = img.resize((self.crop_size[0], self.crop_size[1]))
            label = label.resize((self.crop_size[0], self.crop_size[1]))

            img = np.array(img) / 255
            label = np.array(label)

            if 'cvc' in config['dataset']['train_img_root'] and 'ETIS-LaribPolypDB' not in config['dataset']['test_label_root']:
                # just for cvc start
                label = label[:,:,0]
                # just for cvc end

            img = torch.as_tensor(img)
            label = torch.as_tensor(label)
            if 'Seg' in config['dataset']['train_img_root'] or 'BRATS2015' in config['dataset']['train_img_root']:
                img = img.unsqueeze(0)
            else:
                img = img.permute(2, 0, 1)

            return img.float(), label.long()

    def __len__(self):
        if self.mode == 'train':
            total_img = len(self.train_img_files)
            return total_img
        if self.mode == 'val':
            total_img = len(self.val_img_files)
            return total_img

    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list


# dataset fro Kvasir
class Datareader(Dataset):
    def __init__(self, img_root, label_root, crop_size):
        self.img_files = self.read_file(img_root)
        self.label_files = self.read_file(label_root)
        self.crop_size = crop_size

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        label = Image.open(self.label_files[index])

        img = img.resize((self.crop_size[0], self.crop_size[1]))
        label = label.resize((self.crop_size[0], self.crop_size[1]))

        img = np.array(img) / 255
        label = np.array(label)

        img = torch.as_tensor(img)
        label = torch.as_tensor(label)

        img = img.permute(2, 0, 1)

        return img.float(), label.long()

    def __len__(self):
        total_img = len(self.img_files)
        return total_img

    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

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

        # 初始化图片大小
        self._init_img_size()
        

    def __getitem__(self, index):


        img = Image.open(self.img_files[index]).convert('RGB')
        label = Image.open(self.label_files[index]).convert('L')
        
        name = self.img_files[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'          
        shape = label.size[::-1]
        sample = {'image': img, 'gt': label, 'name': name, 'shape': shape}            
        sample = self.transform(sample)
        img, label = sample['image'],sample['gt']
        img = torch.as_tensor(np.array(img))
        label = torch.as_tensor(np.array(label))        
        return img.float(), label.float()

    def __len__(self):
        total_img = len(self.img_files)
        return total_img

    
    def _init_img_size(self):
        for i in range(self.__len__()):
            img = Image.open(self.img_files[i])
            self.image_size.append(img.size)


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
