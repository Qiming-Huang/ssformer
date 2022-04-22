import torch
import torch.nn as nn


# Helper function to enable loss function to be flexibly used for 
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]

    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


class SymmetricFocalLoss(nn.Module):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=2., epsilon=1e-07):
        super(SymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):

        axis = identify_axis(y_true.size())  
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        #calculate losses separately for each class
        back_ce = torch.pow(1 - y_pred[:,:,:,0], self.gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - self.delta) * back_ce

        fore_ce = torch.pow(1 - y_pred[:,:,:,1], self.gamma) * cross_entropy[:,:,:,1]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss


class AsymmetricFocalLoss(nn.Module):
    """For Imbalanced datasets
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.25
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.25, gamma=2., epsilon=1e-07):
        super(AsymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):

        axis = identify_axis(y_true.size())  
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)
        
	    #calculate losses separately for each class, only suppressing background class
        back_ce = torch.pow(1 - y_pred[:,:,:,0], self.gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - self.delta) * back_ce

        fore_ce = cross_entropy[:,:,:,1]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss


class SymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super(SymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())
        
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1-y_pred), axis=axis)
        fp = torch.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)

        #calculate losses separately for each class, enhancing both classes
        back_dice = (1-dice_class[:,0]) * torch.pow(1-dice_class[:,0], -self.gamma)
        fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -self.gamma) 

        # Average class scores
        loss = torch.mean(torch.stack([back_dice,fore_dice], axis=-1))
        return loss


class AsymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super(AsymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Clip values to prevent division by zero error
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1-y_pred), axis=axis)
        fp = torch.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)

        #calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[:,0]) 
        fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -self.gamma) 

        # Average class scores
        loss = torch.mean(torch.stack([back_dice,fore_dice], axis=-1))
        return loss


class SymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5):
        super(SymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
      symmetric_ftl = SymmetricUnifiedFocalLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)
      symmetric_fl = SymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)
      if self.weight is not None:
        return (self.weight * symmetric_ftl) + ((1-self.weight) * symmetric_fl)  
      else:
        return symmetric_ftl + symmetric_fl


class AsymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.2):
        super(AsymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
      # Obtain Asymmetric Focal Tversky loss
      asymmetric_ftl = AsymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)

      # Obtain Asymmetric Focal loss
      asymmetric_fl = AsymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)

      # return weighted sum of Asymmetrical Focal loss and Asymmetric Focal Tversky loss
      if self.weight is not None:
        return (self.weight * asymmetric_ftl) + ((1-self.weight) * asymmetric_fl)  
      else:
        return asymmetric_ftl + asymmetric_fl


import torch
import torch.nn as nn
import torch.nn.functional as F

# 针对二分类任务的 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
#        pred = nn.Sigmoid()(pred)

        # 展开 pred 和 target,此时 pred.size = target.size = (BatchSize,1) 
        pred = pred.view(-1,1)
        target = target.view(-1,1)

        # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
        pred = torch.cat((1-pred,pred),dim=1)

        # 根据 target 生成 mask，即根据 ground truth 选择所需概率
        # 用大白话讲就是：
        # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0],pred.shape[1]).cuda()
        # 这里的 scatter_ 操作不常用，其函数原型为:
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor. 
        # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用 mask 将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1,1)
        probs = probs.clamp(min=0.0001,max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0],pred.shape[1]).cuda()
        alpha[:,0] = alpha[:,0] * (1-self.alpha)
        alpha[:,1] = alpha[:,1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1,1)

        # 根据 Focal Loss 的公式计算 Loss
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

         # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

 
        