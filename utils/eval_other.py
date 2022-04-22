from medpy.metric.binary import recall as mp_recall
from medpy.metric.binary import dc
import numpy as np
from medpy.metric.binary import precision as mp_precision


def _thresh(img):
    img[img > 0.5] = 1
    img[img <= 0.5] = 0
    return img

def dsc(y_pred, y_true):
  y_pred = _thresh(y_pred)
  y_true = _thresh(y_true)

  return dc(y_pred, y_true)

def iou(y_pred, y_true):
  y_pred = _thresh(y_pred)
  y_true = _thresh(y_true)

  intersection = np.logical_and(y_pred, y_true)
  union = np.logical_or(y_pred, y_true)
  if not np.any(union):
    return 0 if np.any(y_pred) else 1
  
  return intersection.sum() / float(union.sum())

def precision(y_pred, y_true):
  y_pred = _thresh(y_pred).astype(np.int)
  y_true = _thresh(y_true).astype(np.int)

  if y_true.sum() <= 5:
    # when the example is nearly empty, avoid division by 0
    # if the prediction is also empty, precision is 1
    # otherwise it's 0
    return 1 if y_pred.sum() <= 5 else 0

  if y_pred.sum() <= 5:
    return 0.
  
  return mp_precision(y_pred, y_true)

def recall(y_pred, y_true):
    y_pred = _thresh(y_pred).astype(np.int)
    y_true = _thresh(y_true).astype(np.int)

    if y_true.sum() <= 5:
    # when the example is nearly empty, avoid division by 0
    # if the prediction is also empty, recall is 1
    # otherwise it's 0
        return 1 if y_pred.sum() <= 5 else 0

    if y_pred.sum() <= 5:
        return 0.
  
    r = mp_recall(y_pred, y_true)
    return r


