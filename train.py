import torchvision
from models import build
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optmi
import torch.nn.functional as F
from utils.tools import Fmeasure_calu
from utils.my_dataset import CustomDataSet
from utils.loss import *
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision import transforms
import torch
import os
import sys
import numpy as np
import yaml
from tabulate import tabulate
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from warmup_scheduler import GradualWarmupScheduler
from thop import profile


def _thresh(img):
    img[img > 0.5] = 1
    img[img <= 0.5] = 0
    return img

def dsc(y_pred, y_true):
    y_pred = _thresh(y_pred)
    y_true = _thresh(y_true)

    return dc(y_pred, y_true)
np.seterr(divide='ignore', invalid='ignore')
    
np.seterr(divide='ignore', invalid='ignore')
torch.cuda.manual_seed_all(1)
torch.manual_seed(1)

f = open(sys.argv[1])
config = yaml.safe_load(f)

evl_epoch = config['training']['evl_epoch']

# 定义模型
device = config['training']['device']
model = build(model_name=config['model']['model_name'], class_num=config['dataset']['class_num'])
model.to(device)

input = torch.randn(1, 3, 352, 352).to('cuda')
macs, params = profile(model, inputs=(input, ))
print('macs:',macs/1000000000)
print('params:',params/1000000)
logger.info(f"| model |macs:', {macs/1000000000}, 'params:', {params/1000000}|")

# if pretrained 
if config['model']['is_pretrained']:
    model.load_state_dict(torch.load(config['model']['pretrained_path']))
    logger.info("successfully add pretrained model")

train_img_root = config['dataset']['train_img_root']
train_label_root = config['dataset']['train_label_root']

batch_size = config['dataset']['batch_size']
num_workers = config['dataset']['num_workers']
checkpoint_save_path = config['other']['checkpoint_save_path']

# transform_list
Train_transform_list = config['Train_transform_list']
Val_transform_list = config['Val_transform_list']

# training
max_epoch = config['training']['max_epoch']
lr = float(config['training']['lr'])

train_ds = CustomDataSet(train_img_root, train_label_root, transform_list=Train_transform_list)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# criterion = nn.NLLLoss().to(device)
# criterion =nn.CrossEntropyLoss().to(device)
# criterion = nn.BCELoss().to(device)
# criterion = AsymmetricUnifiedFocalLoss()
# criterion = FocalLoss()
# optimizer
optim = optmi.AdamW(model.parameters(), lr=lr)

# scheduler_warmup is chained with schduler_steplr
scheduler_steplr = StepLR(optim, step_size=200, gamma=0.1)
scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=1, after_scheduler=scheduler_steplr)


dataset = ['CVC-300', 'CVC-ColonDB', 'CVC-ClinicDB', 'ETIS-LaribPolypDB', 'Kvasir']
# logger
print(config['other']['logger_path'])
logger.add(config['other']['logger_path'])
# start training
logger.info(f"| start training .... | current model {config['model']['model_name']} |")
logger.info(f"Train_transform_list: | {Train_transform_list}|")
logger.info(f"Val_transform_list: |{Val_transform_list}|")
best_val_dice = [0]
best_loss = [100000]
from_epoch = config['model']['from_epoch']
for epoch in tqdm(range(max_epoch)):
    train_loss = 0
    model.train()
    epoch = epoch + int(from_epoch)
    scheduler_warmup.step(epoch)
    logger.info(f"lr: |{optim.param_groups[0]['lr']}|")
    for idx, (img, label) in tqdm(enumerate(train_loader)):
        model = model.train()
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        out = nn.Sigmoid()(out)
        loss = dice_bce_loss(out, label)
        train_loss += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()

    if (epoch + 1) % 10 == 0:
        logger.critical(f"saving checkpoint at {epoch}")
        torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"{epoch+1}.pth"))

    if train_loss / (idx + 1) < min(best_loss):
        best_loss.append(train_loss / (idx + 1))
        print("train epoch done")
        logger.info(f"| epoch : {epoch} | training done | best loss: {train_loss / (idx + 1)} |")
    else:
        logger.info(f"| epoch : {epoch} | training done | No best loss |")

    if epoch >= evl_epoch:
        model.eval()
        val = []
        model = model.eval()
        
        for i in dataset:
            print("evaluating ", i)
            cot = 0
            from utils.test_dataset import CustomDataSet as test_DataSet
            val_ds = test_DataSet(config['dataset']['test_' + str(i) + '_img'], config['dataset']['test_' + str(i) + '_label'], transform_list=Val_transform_list)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
            total_meanDic = 0
            Thresholds = np.linspace(1, 0, 256)
            with torch.no_grad():
                for idx, (img, label) in tqdm(enumerate(val_loader)):
                    img = img.to(device)
                    label = label.to('cpu')
                    x = model(img)
                    pred = torch.sigmoid(x)
                    pred = F.interpolate(pred, size=(val_ds.image_size[cot][1], val_ds.image_size[cot][0]), mode='bilinear', align_corners=False)
                    cot = cot+1
                    threshold = torch.tensor([0.5]).to(device)
                    pred = (pred > threshold).float() * 1
                    pre_label = pred.squeeze(1).cpu().numpy()
                    true_label = label.squeeze(1).cpu().numpy()
                    threshold_Dice = np.zeros((img.shape[0], len(Thresholds)))

                    for each in range(img.shape[0]):
                        pred = pre_label[each, :].squeeze()
                        label_ = true_label[each, :]
                        label_ = np.array(label_).astype(np.uint8).squeeze()
                        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                        threshold_Dic = np.zeros(len(Thresholds))

                        for j, threshold in enumerate(Thresholds):
                            if j == 0:
                                _, _, _, threshold_Dic[j], _, _ = Fmeasure_calu(pred, label_, threshold)
                                a = threshold_Dic[j]
                            if j == 255:
                                _, _, _, threshold_Dic[j], _, _ = Fmeasure_calu(pred, label_, threshold)
                            if 1 <= j <= 254:
                                threshold_Dic[j] = a

                        threshold_Dice[each, :] = threshold_Dic
                        column_Dic = np.mean(threshold_Dice, axis=0)

                    meanDic = np.mean(column_Dic)
                    total_meanDic = total_meanDic + meanDic
                val.append(total_meanDic / (idx + 1))
                print(val)
        val = np.array(val)
        mean_total = val.mean()
        logger.info(f"| val : {val} | val done |")
        if max(best_val_dice) <= mean_total:
            best_val_dice.append(mean_total)
            table_header = ['Dataset', config['model']['model_name'] + '_Dice', 'UACANet_L_Dice', 'First_Dice']
            table_data = [('CVC-300', str(val[0]), '0.91349', 'None'),
                      ('CVC-ColonDB', str(val[1]), '0.75319', '0.8474'),
                      ('CVC-ClinicDB', str(val[2]), '0.92858', '0.9420'),
                      ('ETIS-LaribPolypDB', str(val[3]), '0.76897', '0.766'),
                      ('Kvasir', str(val[4]), '0.90614', '0.9217'),
                      ('Average', str(val.mean()), '0.853', 'None')]

            logger.info(tabulate(table_data, headers=table_header, tablefmt='psql'))
            torch.save(model.state_dict(), os.path.join(checkpoint_save_path, "best_val.pth"))
        else:
            logger.info(f"| epoch : {epoch} | val done |")

