import torch
from ctypes import c_int
import io
from PIL import Image
from models import build
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optmi
import torch.nn.functional as F
from utils.tools import mean_dice, mean_iou, Fmeasure_calu
from utils.test_dataset import CustomDataSet
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision import transforms
from torchvision.utils import save_image
import torch
import os
import sys
import numpy as np
import yaml
from tabulate import tabulate

np.seterr(divide='ignore', invalid='ignore')

f = open(sys.argv[1])
config = yaml.safe_load(f)

device = config['training']['device']
model = build(model_name=config['model']['model_name'], class_num=config['dataset']['class_num'])

if device == "cpu":
    model.load_state_dict(torch.load(config['test']['checkpoint_save_path']), map_location=torch.device('cpu'))
else:
    model.load_state_dict(torch.load(config['test']['checkpoint_save_path']),strict=False)

model = model.to(device)
model.eval()

train_img_root = config['dataset']['train_img_root']
train_label_root = config['dataset']['train_label_root']


# batch size !!!!
batch_size = 1
num_workers = config['dataset']['num_workers']
checkpoint_save_path = config['other']['checkpoint_save_path']

# training
max_epoch = config['training']['max_epoch']
lr = float(config['training']['lr'])

Test_transform_list = config['Test_transform_list']
torch.cuda.manual_seed_all(1)
torch.manual_seed(1)

#dataset = ['Kvasir']
dataset = ['CVC-300', 'CVC-ColonDB', 'CVC-ClinicDB', 'ETIS-LaribPolypDB','Kvasir']
model = model.eval()
val = []
for i in dataset:
    print(f" predicting {i}")
    val_ds = CustomDataSet(config['dataset']['test_' + str(i) + '_img'], config['dataset']['test_' + str(i) + '_label'], transform_list=Test_transform_list)
    
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    cot = 0
    total_meanDic = 0
    Thresholds = np.linspace(1, 0, 256)
    with torch.no_grad():
        for idx, (img, label) in tqdm(enumerate(val_loader)):
            img = img.to(device)
            label = label.to('cpu')
            x = model(img)
            pred = torch.sigmoid(x)
            pred = F.interpolate(pred, size=(val_ds.image_size[cot][1], val_ds.image_size[cot][0]), mode='bilinear', align_corners=False)

            threshold = torch.tensor([0.5]).to(device)
            pred = (pred > threshold).float() * 1

            pre_label = pred.squeeze(1).cpu().numpy()
            true_label = label.squeeze(1).cpu().numpy()
            threshold_Dice = np.zeros((img.shape[0], len(Thresholds)))

            for each in range(img.shape[0]):
                pred = pre_label[each, :].squeeze()
                label_ = label[each, :]
                label_ = np.array(label_).astype(np.uint8).squeeze()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                threshold_Dic = np.zeros(len(Thresholds))                
                for j, threshold in enumerate(Thresholds):
                    _, _, _, threshold_Dic[j], _, _ = Fmeasure_calu(pred, label_, threshold)
                    
                threshold_Dice[each, :] = threshold_Dic
                column_Dic = np.mean(threshold_Dice, axis=0)

                cot += 1
            meanDic = np.mean(column_Dic)
            total_meanDic = total_meanDic + meanDic
        val.append(total_meanDic / (idx + 1))
        print(val)
	

val = np.array(val)
table_header = ['Dataset', config['model']['model_name']+'_Dice','UACANet_L_Dice','First_Dice']
table_data = [('CVC-300',str(val[0]), '0.91349','None'),
			 ('CVC-ColonDB',str(val[1]),'0.75319','0.8474'),
			('CVC-ClinicDB',str(val[2]),'0.92858','0.9420' ),
			('ETIS-LaribPolypDB',str(val[3]),'0.76897','0.766'),
			('Kvasir',str(val[4]),'0.90614','0.9217'),
			('Average',str(val.mean()),'0.853','None'),]
			
print(tabulate(table_data, headers=table_header, tablefmt='psql'))




