import numpy as np
import os
import time
import scipy.io as scio
import random
import itertools
from scipy import misc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import math
from scipy.ndimage.filters import gaussian_filter, median_filter
import multiprocessing

#from DenseEncoderIAMUShapeDecoder import * 
from baseline_faircmp import * 

from custom import *

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
    
def processing_func(pred_dir, i, post_dilation_iter=2):
    matd = scio.loadmat(pred_dir+str(i)+'.mat')
    s = matd['s']
    c = matd['c']
    lab_img = post_proc(s-c, post_dilation_iter=post_dilation_iter).squeeze()
    scio.savemat(pred_dir+str(i)+'.mat', {'instance':lab_img, 's':s, 'c':c})
    print(str(i)+' th obj finished!')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.set_num_threads(8)

n_pred_labels_type = 1
n_pred_labels_bnd = 1
input_modalities = 3
batch_sz = 4
MAX_epoch = 600
save_cp_after_n_epoch = 20
aug_list = ['ori', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_w']

ndata = 16
ntrain = 4
kfold_train_idx, kfold_val_idx = kfold_list(ndata, ntrain)

# Image Net mean and std
norm_mean=[0.485, 0.456, 0.406]
norm_std=[0.229, 0.224, 0.225]
valdata_filefold = '/home/cong/workplace/kumar'
imgs = np.load(os.path.join(valdata_filefold, 'data_after_stain_norm_ref1.npy'))
imgs = imgs.transpose([0, 3, 1, 2]).astype(np.float32)
ndata, mod, h0, w0 = imgs.shape
for imod in range(mod):
    imgs[:, imod] = (imgs[:, imod]/255.0 - norm_mean[imod])/norm_std[imod]

net_name = 'baseline_'+str(ntrain)+'fold_epoch'+str(MAX_epoch)

for itrain in list(range(0, ntrain)):
    dir_checkpoint = 'weight/'+net_name+'/rnd_'+str(itrain)+'/'
    dir_savefile = 'val/'+net_name+'/rnd_'+str(itrain)+'/'
    confirm_loc(dir_savefile)
    net = UNet(input_modalities, n_pred_labels_type, n_pred_labels_bnd).cuda()
    sigmoid = nn.Sigmoid()
    snapshot_list = list(range(save_cp_after_n_epoch, MAX_epoch+1, save_cp_after_n_epoch))
    with torch.no_grad():
        for isnapshot in snapshot_list:
            confirm_loc(dir_savefile + 'snapshot_' + str(isnapshot)+ '/')
            weight_loc = dir_checkpoint+'model_of_'+str(isnapshot)+'.pth'
            if not(os.path.exists(weight_loc)):
                continue
            weight = torch.load(weight_loc, map_location=lambda storage, loc: storage.cuda())
            target_dict = net.state_dict()
            source_dict = weight
            source_dict2 = {}
            for k,v in source_dict.items():
                if k in target_dict:
                    source_dict2.update({k:v})
            target_dict.update(source_dict2)
            net.load_state_dict(target_dict)
            net.eval()

            for i_data in kfold_val_idx[itrain]:
                print('Processing ' + str(i_data) + ' obj in ' + str(isnapshot))
                d = imgs[i_data].copy()
                mod, h, w = d.shape
                finalsout = np.zeros([n_pred_labels_type, h, w], dtype=np.float32)
                finalcout = np.zeros([n_pred_labels_bnd, h, w], dtype=np.float32)
                aug_ncount = 0

                d0 = torch.FloatTensor(d).unsqueeze(dim=0)
                #szs = [320, 288, 256, 224, 192]
                #c_sz = [160, 144, 128, 112, 96]
                szs = [256]
                c_sz = [128]
                # 10, 20, 10 / 8, 16, 8 / 6, 12, 6
                # 7 * 20 * 8 = 1120
                # 8 * 16 * 8 = 1024
                # 11 * 12 * 8 = 1056
                print(d.shape)
                for isz, sz in enumerate(szs):
                    patches_d, ibs, shs, sws = extract_patches(d0, sz, c_sz[isz])
                    # patch_weight_map: Nc * Hp * Wp
                    center_x_map = torch.FloatTensor(list(range(0, szs[isz], 1))).view(szs[isz], 1)
                    center_x_map = center_x_map.expand(-1, szs[isz])
                    center_y_map = center_x_map.t()
                    patch_weight_map = ((center_x_map - szs[isz]//2).pow(2) + (center_y_map - szs[isz]//2).pow(2)).sqrt()
                    patch_weight_map = (1-patch_weight_map / patch_weight_map.max()).pow(2)
                    patch_weight_map = patch_weight_map.clamp(min = 1e-3, max = 1.0-1e-3)
                    patch_weight_map_seg = patch_weight_map.unsqueeze(dim=0).expand(n_pred_labels_type, -1, -1)
                    patch_weight_map_bnd = patch_weight_map.unsqueeze(dim=0).expand(n_pred_labels_bnd, -1, -1)

                    for type_aug in aug_list:
                        print('Processing {0:d} obj, sz {1:d}, aug{2:d} type '.format(i_data, sz, aug_ncount)+type_aug)
                        aug_ncount += 1
                        # TTA
                        if type_aug == 'ori':
                            tta_d = patches_d.clone()
                        elif type_aug == 'rot90':
                            tta_d = patches_d.rot90(1, dims=(2,3))
                        elif type_aug == 'rot180':
                            tta_d = patches_d.rot90(2, dims=(2,3))
                        elif type_aug == 'rot270':
                            tta_d = patches_d.rot90(3, dims=(2,3))
                        elif type_aug == 'flip_h':
                            tta_d = patches_d.flip(2)
                        elif type_aug == 'flip_w':
                            tta_d = patches_d.flip(3)

                        spred = torch.zeros(tta_d.shape[0], n_pred_labels_type, tta_d.shape[2], tta_d.shape[3])
                        cpred = torch.zeros(tta_d.shape[0], n_pred_labels_bnd, tta_d.shape[2], tta_d.shape[3])

                        for sb in list(range(0, tta_d.shape[0], batch_sz)):
                            eb = np.min([sb+batch_sz, tta_d.shape[0]])
                            input_d = tta_d[sb:eb]
                            if len(input_d.shape) == 3:
                                input_d = input_d.unsqueeze(dim=0)
                            outs = net(input_d.cuda())
                            sout = outs[0]
                            cout = outs[5]
                            spred[sb:eb] = sigmoid(sout).data.cpu()
                            cpred[sb:eb] = sigmoid(cout).data.cpu()
                        # Inverse TTA
                        if type_aug == 'rot90':
                            spred = spred.rot90(3, dims=(len(spred.shape)-2,len(spred.shape)-1))
                            cpred = cpred.rot90(3, dims=(len(cpred.shape)-2,len(cpred.shape)-1))
                        elif type_aug == 'rot180':
                            spred = spred.rot90(2, dims=(len(spred.shape)-2,len(spred.shape)-1))
                            cpred = cpred.rot90(2, dims=(len(cpred.shape)-2,len(cpred.shape)-1))
                        elif type_aug == 'rot270':
                            spred = spred.rot90(1, dims=(len(spred.shape)-2,len(spred.shape)-1))
                            cpred = cpred.rot90(1, dims=(len(cpred.shape)-2,len(cpred.shape)-1))
                        elif type_aug == 'flip_h':
                            spred = spred.flip(len(spred.shape)-2)
                            cpred = cpred.flip(len(cpred.shape)-2)
                        elif type_aug == 'flip_w':
                            spred = spred.flip(len(spred.shape)-1)
                            cpred = cpred.flip(len(cpred.shape)-1)

                        spred_map = reconstruct_from_patches_weightedall(spred, ibs, shs, sws, sz, c_sz[isz], d0.shape[0], 1, d0.shape[2], d0.shape[3], 1).squeeze().numpy()
                        cpred_map = reconstruct_from_patches_weightedall(cpred, ibs, shs, sws, sz, c_sz[isz], d0.shape[0], 1, d0.shape[2], d0.shape[3], 1).squeeze().numpy()

                        finalsout += spred_map
                        finalcout += cpred_map

                finalsout /= aug_ncount
                finalcout /= aug_ncount
                
                scio.savemat(dir_savefile + 'snapshot_' + str(isnapshot)+ '/' + str(i_data)+'.mat', {'s':finalsout.astype(np.float32), 'c':finalcout.astype(np.float32)})     
