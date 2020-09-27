import os
import numpy as np
import torch

def confirm_loc(loc):
    isExists=os.path.exists(loc)
    if not isExists:
        os.makedirs(loc)

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
