from patch_dense_net import UNet
import numpy as np
import os
import glob
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
from skimage.morphology import label
from scipy import ndimage
from scipy.ndimage import zoom
from custom import confirm_loc, extract_patches, reconstruct_from_patches_weightedall

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.set_num_threads(8)

# imageneet mean and std
norm_mean=[0.485, 0.456, 0.406]
norm_std=[0.229, 0.224, 0.225]

n_pred_labels = 1
input_modalities = 3
valsegcenterpred_filefold = '../tafe/train/tafe_4fold_epoch600/'

data_filefold = '/home/cong/workplace/kumar'
imgs = np.load(os.path.join(data_filefold, 'data_after_stain_norm_ref1.npy')) # .mat of train set
imgs = imgs.transpose([0, 3, 1, 2]).astype(np.float32)
ndata, mod, h0, w0 = imgs.shape
for imod in range(mod):
    imgs[:, imod] = (imgs[:, imod]/255.0 - norm_mean[imod])/norm_std[imod]

maxpad = 176//2
pred_sz_type_list = ['48', '176']
dilation_s = 0
dilation_c = 2
loss_type = 'focalloss'
#aug_list = ['ori', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_w']
aug_list = ['ori']

for ifold in range(4):

    list_dmat = scio.loadmat(valsegcenterpred_filefold+'/patchnet_trainset_list_ifold'+str(ifold)+'.mat');
    valset = list_dmat['valset']
    valset = valset-1

    for dilation_rate in [2]:
        for ths in [5]:
            for pred_sz_type in pred_sz_type_list:
                if pred_sz_type == '48':
                    low_sz_ths = 1
                    high_sz_ths = 48//2
                    cursz = 48
                    batchsz = 256
                elif pred_sz_type == '176':
                    low_sz_ths = 48//2
                    high_sz_ths = 1000
                    cursz = 176
                    batchsz = 24
                else:
                    print('error pred_sz_type')

                valimgs = np.pad(imgs, ((0,0), (0,0), (maxpad, maxpad), (maxpad, maxpad)), 'reflect')
                
                print(cursz)
                for irnd, iseed in enumerate([12345]):
                    dir_checkpoint = './weight/noaug_nodrop_'+loss_type+'_iouthsp'+str(ths)+'_dilation_'+str(dilation_rate)+'_sz'+str(cursz)+'_ds'+str(dilation_s)+'_dc'+str(dilation_c)+'_ifold'+str(ifold)+'/rnd_'+str(iseed)+'/'
                    print(dir_checkpoint)
                    dir_savefile = './val/noaug_nodrop_'+loss_type+'_iouthsp'+str(ths)+'_dilation_'+str(dilation_rate)+'_sz'+str(cursz)+'_ds'+str(dilation_s)+'_dc'+str(dilation_c)+'_ifold'+str(ifold)+'_wtta/rnd_'+str(iseed)+'/'
                    confirm_loc(dir_savefile)

                    net = UNet(input_modalities, n_pred_labels*2, n_pred_labels).cuda()

                    sigmoid = nn.Sigmoid()
                    snapshot_list = glob.glob(dir_checkpoint+'*.pth')
                    with torch.no_grad():

                        for isnapshot_name in snapshot_list:

                            isnapshot = int(isnapshot_name.split('/')[-1].split('.')[0].split('_')[-1])
                            confirm_loc(dir_savefile + 'snapshot_' + str(isnapshot)+ '/')
                            weight_loc = isnapshot_name
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

                            for i_data in valset[irnd]:
                                dmat = scio.loadmat(valsegcenterpred_filefold+'pred_res_'+str(i_data)+'_withpostproc.mat')
                                tmp_instance = dmat['instance_nodilation']
                                tmp_instance = np.pad(tmp_instance, ((maxpad, maxpad), (maxpad, maxpad)), 'constant', constant_values=0)
                                tmps = np.pad(dmat['s'], ((maxpad, maxpad), (maxpad, maxpad)), 'constant', constant_values=0)
                                tmpc = np.pad(dmat['c'], ((maxpad, maxpad), (maxpad, maxpad)), 'constant', constant_values=0)
                                cset = np.unique(tmp_instance[tmp_instance>0])
                                d = valimgs[i_data].copy()
                                mod, h, w = d.shape
                                
                                # pixels with probability larger than 0.5 are considered as foreground pixels
                                instances = []
                                instances.append(np.zeros([1, h, w])+0.5)

                                n_suitable_instance = 0
                                patches_cursz_img = []
                                patches_cursz_sin = []
                                patches_cursz_cin = []
                                sxs = []
                                sys = []
                                exs = []
                                eys = []
                                halfszs = []
                                for ic in cset:
                                    icmap = tmp_instance==ic
                                    if dilation_rate > 0:
                                        icmap = ndimage.morphology.binary_dilation(icmap, iterations=dilation_rate)
                                    if dilation_s > 0:
                                        dicmap_s = ndimage.morphology.binary_dilation(icmap, iterations=dilation_s)
                                    elif dilation_s == 0:
                                        dicmap_s = icmap.copy()
                                    else:
                                        dicmap_s = np.ones(icmap.shape)
                                        
                                    if dilation_c > 0:
                                        dicmap_c = ndimage.morphology.binary_dilation(icmap, iterations=dilation_c)
                                    elif dilation_c == 0:
                                        dicmap_c = icmap.copy()
                                    else:
                                        dicmap_c = np.ones(icmap.shape)
                                        
                                    icx, icy = np.nonzero(icmap)
                                    maxx = icx.max()
                                    maxy = icy.max()
                                    minx = icx.min()
                                    miny = icy.min()
                                    mx = np.round((maxx+minx)/2)
                                    my = np.round((maxy+miny)/2)
                                    halfsz = (np.max([(maxx-minx)/2, (maxy-miny)/2, 8])+12).astype(np.int16)
                                    sx = np.round(mx - halfsz).astype(np.int16)
                                    sy = np.round(my - halfsz).astype(np.int16)
                                    ex = np.round(mx + halfsz + 1).astype(np.int16)
                                    ey = np.round(my + halfsz + 1).astype(np.int16)
                                    
                                    if halfsz>=low_sz_ths and halfsz<high_sz_ths:
                                        n_suitable_instance += 1
                                        patch_img = d[:, sx:ex, sy:ey].astype(np.float32)
                                        patch_sin = tmps[sx:ex, sy:ey].astype(np.float32) * dicmap_s[sx:ex, sy:ey].astype(np.float32)
                                        patch_cin = tmpc[sx:ex, sy:ey].astype(np.float32) * dicmap_c[sx:ex, sy:ey].astype(np.float32)
                                        cursz_img = zoom(patch_img, [1, cursz/(halfsz*2+1).astype(np.float64), cursz/(halfsz*2+1).astype(np.float64)], order=1)
                                        cursz_sin = zoom(patch_sin, [cursz/(halfsz*2+1).astype(np.float64), cursz/(halfsz*2+1).astype(np.float64)], order=1)
                                        cursz_cin = zoom(patch_cin, [cursz/(halfsz*2+1).astype(np.float64), cursz/(halfsz*2+1).astype(np.float64)], order=1)

                                        cursz_img = torch.FloatTensor(cursz_img.astype(np.float32))
                                        cursz_sin = torch.FloatTensor(cursz_sin.astype(np.float32))
                                        cursz_cin = torch.FloatTensor(cursz_cin.astype(np.float32))

                                        patches_cursz_img.append(cursz_img.unsqueeze(dim=0))
                                        patches_cursz_sin.append(cursz_sin.unsqueeze(dim=0).unsqueeze(dim=1))
                                        patches_cursz_cin.append(cursz_cin.unsqueeze(dim=0).unsqueeze(dim=1))
                                        
                                        sxs.append(sx)
                                        sys.append(sy)
                                        exs.append(ex)
                                        eys.append(ey)
                                        halfszs.append(halfsz)

                                npatches = len(patches_cursz_img)
                                if npatches>0:
                                    patches_img = torch.cat(tuple(patches_cursz_img), dim=0)
                                    patches_sin = torch.cat(tuple(patches_cursz_sin), dim=0)
                                    patches_cin = torch.cat(tuple(patches_cursz_cin), dim=0)
                                                                    
                                    for sipatch in list(range(0, npatches, batchsz)):
                                        eipatch = min(sipatch+batchsz, npatches)
                                        batch_img = patches_img[sipatch:eipatch]
                                        batch_sin = patches_sin[sipatch:eipatch]
                                        batch_cin = patches_cin[sipatch:eipatch]
                                        
                                        n_inbatch = batch_img.shape[0]
                                        
                                        batch_scin = torch.cat((batch_sin, batch_cin), dim=1)

                                        tta_img = []
                                        tta_scin = []
                                        for type_aug in aug_list:
                                            # TTA
                                            if type_aug == 'ori':
                                                tta_img.append(batch_img.clone())
                                                tta_scin.append(batch_scin.clone())
                                            elif type_aug == 'rot90':
                                                tta_img.append(batch_img.rot90(1, dims=(2, 3)))
                                                tta_scin.append(batch_scin.rot90(1, dims=(2, 3)))
                                            elif type_aug == 'rot180':
                                                tta_img.append(batch_img.rot90(2, dims=(2, 3)))
                                                tta_scin.append(batch_scin.rot90(2, dims=(2, 3)))
                                            elif type_aug == 'rot270':
                                                tta_img.append(batch_img.rot90(3, dims=(2, 3)))
                                                tta_scin.append(batch_scin.rot90(3, dims=(2, 3)))
                                            elif type_aug == 'flip_h':
                                                tta_img.append(batch_img.flip(2))
                                                tta_scin.append(batch_scin.flip(2))
                                            elif type_aug == 'flip_w':
                                                tta_img.append(batch_img.flip(3))
                                                tta_scin.append(batch_scin.flip(3))

                                        tta_img = torch.cat(tuple(tta_img), dim=0)
                                        tta_scin = torch.cat(tuple(tta_scin), dim=0)
                                                                            
                                        spred = net(tta_img.cuda(), tta_scin.cuda())
                                        # (n*naug) * H * W
                                        if isinstance(spred, tuple):
                                            spred = F.sigmoid(spred[0]).data.cpu().squeeze()
                                        else:
                                            spred = F.sigmoid(spred).data.cpu().squeeze()                                    
                                        aug_spred = np.zeros([n_inbatch, cursz, cursz], dtype=np.float32)
                                        for iaug, type_aug in enumerate(aug_list):
                                            idx_iaug_s = n_inbatch*iaug
                                            idx_iaug_e = n_inbatch*(iaug+1)
                                            # Inverse TTA
                                            if type_aug == 'ori':
                                                aug_spred += spred[idx_iaug_s:idx_iaug_e].numpy()
                                            elif type_aug == 'rot90':
                                                aug_spred += spred[idx_iaug_s:idx_iaug_e].rot90(3, dims=(1, 2)).numpy()
                                            elif type_aug == 'rot180':
                                                aug_spred += spred[idx_iaug_s:idx_iaug_e].rot90(2, dims=(1, 2)).numpy()
                                            elif type_aug == 'rot270':
                                                aug_spred += spred[idx_iaug_s:idx_iaug_e].rot90(1, dims=(1, 2)).numpy()
                                            elif type_aug == 'flip_h':
                                                aug_spred += spred[idx_iaug_s:idx_iaug_e].flip(1).numpy()
                                            elif type_aug == 'flip_w':
                                                aug_spred += spred[idx_iaug_s:idx_iaug_e].flip(2).numpy()
                                        aug_spred /= len(aug_list)

                                        for ibatch in list(range(n_inbatch)):
                                            sx = sxs[ibatch+sipatch]
                                            sy = sys[ibatch+sipatch]
                                            ex = exs[ibatch+sipatch]
                                            ey = eys[ibatch+sipatch]
                                            halfsz = halfszs[ibatch+sipatch]

                                            probmap_instances = np.zeros([1, h, w])
                                            probmap_instances[:, sx:ex, sy:ey] = zoom(aug_spred[ibatch], [(halfsz*2+1).astype(np.float64)/cursz, (halfsz*2+1).astype(np.float64)/cursz], order=1)
                                            instances.append(probmap_instances.astype(np.float32))

                                    print('Finished Processing ' + str(i_data) + 'th image in ' + str(isnapshot) + ', including ' + str(len(instances)) + ' objs(sz: '+str(cursz)+')')

                                    instances = np.concatenate(instances, axis=0)
                                    maxprob_instances = np.max(instances, axis=0)
                                    idx_instances = np.argmax(instances, axis=0)

                                    scio.savemat(dir_savefile+'snapshot_'+str(isnapshot)+'/'+str(i_data)+'.mat', {'instance':idx_instances[maxpad:(h0+maxpad), maxpad:(w0+maxpad)], 'instance_ori':tmp_instance[maxpad:(h0+maxpad), maxpad:(w0+maxpad)]})
                                    