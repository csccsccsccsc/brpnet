import os
import numpy as np
import torch
import torch.nn.functional as F

def confirm_loc(loc):
    isExists=os.path.exists(loc)
    if not isExists:
        os.makedirs(loc)

def kfold_list(n, K, seed=123):
    rnd_state = np.random.RandomState(seed)
    dataset_rnd_sort = np.array(list(range(n)))
    rnd_state.shuffle(dataset_rnd_sort)
    idxlist = list(range(0, n, 1))
    val_idx = [list(range(i*(n//K), (i+1)*(n//K))) for i in range(0, K, 1)]
    train_idx = [list(range(0, i*int(n//K), 1))+list(range((i+1)*(n//K), n, 1)) for i in range(0, K, 1)]
    if val_idx[-1][-1] < n-1:
        val_idx[-1] = val_idx[-1]+list(range(val_idx[-1][-1]+1, n, 1))
        train_idx[-1] = list(range(0, (K-1)*(n//K)))
    val_from_list = []
    train_from_list = []
    for ival, itrain in zip(val_idx, train_idx):
        val_from_list.append(dataset_rnd_sort[ival])
        train_from_list.append(dataset_rnd_sort[itrain])
    return train_from_list, val_from_list

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']
    
def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def replace_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def extract_patches(img, patch_sz, stride):
    b, c,  h, w = img.shape
    padding_h = (patch_sz - stride)//2
    padding_w = (patch_sz - stride)//2
    padding_h_img = int(np.ceil(h/stride)*stride - h) // 2
    padding_w_img = int(np.ceil(w/stride)*stride - w) // 2
    pad_img = F.pad(img, (padding_w_img + padding_w, padding_w_img + padding_w, padding_h_img + padding_h, padding_h_img + padding_h), mode='reflect')
    _, _, h_pad, w_pad = pad_img.shape
    patches = []
    ibs = []
    shs = []
    sws = []
    for ib in list(range(b)):
        for sh in list(range(padding_h, padding_h+h+padding_h_img*2, stride)):
            for sw in list(range(padding_w, padding_w+w+padding_w_img*2, stride)):
                tmp_p = pad_img[ib, :, (sh-padding_h):(sh+padding_h+stride), (sw-padding_w):(sw+padding_w+stride)].unsqueeze(dim=0)
                patches.append(tmp_p)
                ibs.append(ib)
                shs.append(sh)
                sws.append(sw)
    patches = torch.cat(tuple(patches), dim=0)
    return patches, ibs, shs, sws

def reconstruct_from_patches_weightedall(patches, ibs, shs, sws, patch_sz, stride, b, c, h, w, patches_weight_map):
    padding_h = (patch_sz - stride)//2
    padding_w = (patch_sz - stride)//2
    padding_h_img = int(np.ceil(h/stride)*stride - h) // 2
    padding_w_img = int(np.ceil(w/stride)*stride - w) // 2
    img_rc = torch.zeros(b, c, h+2*padding_h_img+2*padding_h, w+2*padding_w_img+2*padding_w)
    ncount = torch.zeros(b, c, h+2*padding_h_img+2*padding_h, w+2*padding_w_img+2*padding_w)
    #if len(patches_weight_map.shape)==3:
    #    patches_weight_map = patches_weight_map.unsqueeze(dim=0)
    ipatches = 0
    for ipatches in list(range(patches.shape[0])):
        ib = ibs[ipatches]
        sh = shs[ipatches]
        sw = sws[ipatches]
        img_rc[ib, :, (sh-padding_h):(sh+padding_h+stride), (sw-padding_w):(sw+padding_w+stride)] += patches[ipatches] * patches_weight_map
        ncount[ib, :, (sh-padding_h):(sh+padding_h+stride), (sw-padding_w):(sw+padding_w+stride)] += patches_weight_map
    img_rc_norm =  img_rc / ncount
    img_rc_norm = img_rc_norm[:, :, (padding_h_img+padding_h):(padding_h_img+padding_h+h), (padding_w_img+padding_w):(padding_w_img+padding_w+w)]
    return img_rc_norm
    
