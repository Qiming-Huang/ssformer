import models
import utils
# from utils import ISIC2018
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optmi
import os
import pdb
import numpy as np



def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category IoU, shape (num_classes, ).
    """

    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, iou


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category dice, shape (num_classes, ).
    """

    all_acc, acc, dice = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, dice


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category evalution metrics, shape (num_classes, ).
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes, ignore_index,
                                                     label_map,
                                                     reduce_zero_label)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    ret_metrics = [all_acc, acc]
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics

def get_confusion_matrix(pred_label, label, num_classes, ignore_index):
    """Intersection over Union
       Args:
           pred_label (np.ndarray): 2D predict map
           label (np.ndarray): label 2D label map
           num_classes (int): number of categories
           ignore_index (int): index ignore in evaluation
       """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    n = num_classes
    inds = n * label + pred_label
    mat = np.bincount(inds, minlength=n**2).reshape(n, n)
    return mat

def legacy_mean_dice(results, gt_seg_maps, num_classes, ignore_index):
    
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_mat = np.zeros((num_classes, num_classes), dtype=np.float)
    for i in range(num_imgs):
        mat = get_confusion_matrix(results[i], gt_seg_maps[i], num_classes, ignore_index=ignore_index)
        total_mat += mat
    # mat = get_confusion_matrix(results, gt_seg_maps, num_classes, ignore_index=ignore_index)
    # total_mat = mat
    all_acc = np.diag(total_mat).sum() / total_mat.sum()
    acc = np.diag(total_mat) / total_mat.sum(axis=1)
    dice = 2 * np.diag(total_mat) / (total_mat.sum(axis=1) + total_mat.sum(axis=0))

    return all_acc, acc, dice
    
# This func is deprecated since it's not memory efficient
def legacy_mean_iou(results, gt_seg_maps, num_classes, ignore_index):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    # total_mat = np.zeros((num_classes, num_classes), dtype=np.float)
    # for i in range(num_imgs):
    #     mat = get_confusion_matrix(
    #         results[i], gt_seg_maps[i], num_classes, ignore_index=ignore_index)
    #     total_mat += mat
    mat = get_confusion_matrix(results, gt_seg_maps, num_classes, ignore_index=ignore_index)
    total_mat = mat
    all_acc = np.diag(total_mat).sum() / total_mat.sum()
    acc = np.diag(total_mat) / total_mat.sum(axis=1)
    iou = np.diag(total_mat) / (
        total_mat.sum(axis=1) + total_mat.sum(axis=0) - np.diag(total_mat))

    return all_acc, acc, iou


def Fmeasure_calu(pred, gt, threshold):
    if threshold > 1:
        threshold = 1

    Label3 = np.zeros_like(gt)
    Label3[pred >= threshold] = 1

    NumRec = np.sum(Label3 == 1)
    NumNoRec = np.sum(Label3 == 0)

    LabelAnd = (Label3 == 1) & (gt == 1)
    NumAnd = np.sum(LabelAnd == 1)
    num_obj = np.sum(gt)
    num_pred = np.sum(Label3)

    FN = num_obj - NumAnd
    FP = NumRec - NumAnd
    TN = NumNoRec - FN

    if NumAnd == 0:
        PreFtem = 0
        RecallFtem = 0
        FmeasureF = 0
        Dice = 0
        SpecifTem = 0
        IoU = 0

    else:
        IoU = NumAnd / (FN + NumRec)
        PreFtem = NumAnd / NumRec
        RecallFtem = NumAnd / num_obj
        SpecifTem = TN / (TN + FP)
        Dice = 2 * NumAnd / (num_obj + num_pred)
        FmeasureF = ((2.0 * PreFtem * RecallFtem) / (PreFtem + RecallFtem))

    return PreFtem, RecallFtem, SpecifTem, Dice, FmeasureF, IoU


class Colorize:
    def __init__(self, n):
        self.cmap = self.colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])#array->tensor

    def colormap(self, n):
        cmap=np.zeros([n, 3]).astype(np.uint8)
        cmap[0,:] = np.array([ 0,  0,  0])
        cmap[1,:] = np.array([244, 35,232])
        cmap[2,:] = np.array([ 70, 70, 70])
        cmap[3,:] = np.array([ 102,102,156])
        cmap[4,:] = np.array([ 190,153,153])
        cmap[5,:] = np.array([ 153,153,153])
    
        cmap[6,:] = np.array([ 250,170, 30])
        cmap[7,:] = np.array([ 220,220,  0])
        cmap[8,:] = np.array([ 107,142, 35])
        cmap[9,:] = np.array([ 152,251,152])
        cmap[10,:] = np.array([ 70,130,180])
    
        cmap[11,:] = np.array([ 220, 20, 60])
        cmap[12,:] = np.array([ 119, 11, 32])
        cmap[13,:] = np.array([ 0,  0,142])
        cmap[14,:] = np.array([  0,  0, 70])
        cmap[15,:] = np.array([  0, 60,100])
    
        cmap[16,:] = np.array([  0, 80,100])
        cmap[17,:] = np.array([  0,  0,230])
        cmap[18,:] = np.array([ 255,  0,  0])
        
        return cmap
 
    def __call__(self, gray_image):
        size = gray_image.size()#这里就是上文的output
        color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)
 
        for label in range(0, len(self.cmap)):
            mask = gray_image == label
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]
 
        return color_image


from torchvision import transforms
from torchvision.utils import save_image
class ImageSaver():
    def __init__(self):
        self.img = None
        self.label = None

        self.to_Tensor = transforms.Compose([
            transforms.ToTensor()])  
    
        self.to_PIL = transforms.ToPILImage()   

    def save(self, pred, label, b):
        """
        img : tensor (B, C, W, H)
        label: tensor (B, W, H)
        b : batch size
        """
        cot = 0
        for each in range(b):
            img = pred[each, :]
            mask = label[each, :]

            # img = img.resize((val_ds.image_size[cot][0], val_ds.image_size[cot][1]))
            # mask = mask.resize((val_ds.image_size[cot][0], val_ds.image_size[cot][1]))

            # img = self.to_Tensor(img)
            # mask = self.to_Tensor(mask)

            save_image(img, "./predict_images/cvc-300/"+str(cot)+".png")
            save_image(mask, "./predict_labels/CVC-300/"+str(cot)+".png")

            cot += 1        