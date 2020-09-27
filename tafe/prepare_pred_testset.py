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

from DenseEncoderIAMUShapeDecoder import * 
#from baseline_faircmp import * 
from custom import *

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
torch.set_num_threads(8)

n_pred_labels_type = 1
n_pred_labels_bnd = 1
input_modalities = 3
batch_sz = 4
MAX_epoch = 600
save_cp_after_n_epoch = 20
aug_list = ['ori', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_w']

# Image Net mean and std
norm_mean=[0.485, 0.456, 0.406]
norm_std=[0.229, 0.224, 0.225]
data_filefold = '/home/cong/workplace/kumar'

# Predictions of the whole test set
imgs = np.load(os.path.join(data_filefold, 'valdata_after_stain_norm_mm_ref1.npy'))
imgs = imgs.transpose([0, 3, 1, 2]).astype(np.float32)
ndata, mod, h0, w0 = imgs.shape
for imod in range(mod):
    imgs[:, imod] = (imgs[:, imod]/255.0 - norm_mean[imod])/norm_std[imod]
ntrain = 4

#net_name = 'baseline'+'_'+str(ntrain)+'fold_epoch'+str(MAX_epoch)
net_name = 'tafe'+'_'+str(ntrain)+'fold_epoch'+str(MAX_epoch)

# Select From Trained Models
# TAFE:
snapshot_lists = [[27], [4], [9], [8]]
# Baseline:
# snapshot_lists = [[10], [4], [20], [15]]

for itrain in list(range(0, ntrain)):
    dir_checkpoint = 'weight/'+net_name+'/rnd_'+str(itrain)+'/'
    dir_savefile = 'test/'+net_name+'/rnd_'+str(itrain)+'/'
    confirm_loc(dir_savefile)
    net = UNet(input_modalities, n_pred_labels_type, n_pred_labels_bnd).cuda()
    sigmoid = nn.Sigmoid()

    snapshot_list = snapshot_lists[itrain]
    with torch.no_grad():
        for isnapshot in snapshot_list:
            confirm_loc(dir_savefile + 'snapshot_' + str(isnapshot*save_cp_after_n_epoch)+ '/')
            weight_loc = dir_checkpoint+'model_of_'+str(isnapshot*save_cp_after_n_epoch)+'.pth'
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

            for i_data in range(ndata):
                print('Processing ' + str(i_data) + ' obj in ' + str(isnapshot*save_cp_after_n_epoch))
                d = imgs[i_data].copy()
                mod, h, w = d.shape
                finalsout = np.zeros([n_pred_labels_type, h, w], dtype=np.float32)
                finalcout = np.zeros([n_pred_labels_bnd, h, w], dtype=np.float32)
                aug_ncount = 0

                d0 = torch.FloatTensor(d).unsqueeze(dim=0)

                szs = [256]
                c_sz = [128]

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
                
                scio.savemat(dir_savefile + 'snapshot_' + str(isnapshot*save_cp_after_n_epoch)+ '/' + str(i_data)+'.mat', {'s':finalsout.astype(np.float32), 'c':finalcout.astype(np.float32)})     