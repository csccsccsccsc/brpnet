import numpy as np
import os
import time
import math

import torch
import torch.nn.functional as F

def dice_loss_perimg(input, target, eps=1e-7):
    b = input.shape[0]
    iflat = input.contiguous().view(b, -1)
    tflat = target.contiguous().view(b, -1)
    intersection = (iflat * tflat).sum(dim=1)
    return (1 - ((2. * intersection + eps) / (iflat.pow(2).sum(dim=1) + tflat.pow(2).sum(dim=1) + eps))).mean()

def balance_bce_loss(input, target):
    L0 = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    return 0.5*((L0*target).sum()/target.sum()+(L0*(1-target)).sum()/(1-target).sum())

def smooth_truncated_loss(p, t, ths, if_reduction=True, if_balance=False):
    n_log_pt = F.binary_cross_entropy_with_logits(p, t, reduction='none')
    pt = (-n_log_pt).exp()
    L = torch.where(pt>=ths, n_log_pt, -math.log(ths)+0.5*(1-pt.pow(2)/(ths**2)))
    if if_reduction:
        if if_balance:
            return 0.5*((L*t).sum()/t.sum() + (L*(1-t)).sum()/(1-t).sum())
        else:
            return L.mean()
    else:
        return L

def focal_loss(inputs, targets, gamma=2.0):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = (1-pt).pow(gamma) * BCE_loss
    return F_loss.mean()

def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
def replace_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    