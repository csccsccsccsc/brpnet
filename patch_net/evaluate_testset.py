import os
import numpy as np
import scipy.io as scio

from metrics import get_fast_aji, remap_label
from custom import *

ndata = 14
save_cp_after_n_epoch = 1
size_list = ['48', '176']

if_save = True

losstype = 'focalloss'
iou_ths = 5 # 0.5
idilation = 2


pred_root_dir = '/data0/cong/workplace/brp_net/patch_net/test'
gts = np.load('/home/cong/workplace/kumar/valgt.npy').astype(np.uint16)

ajis = []
for i, gt in enumerate(gts):
    spred = np.zeros(gt.shape, dtype=np.uint16)
    for isz, cur_sz in enumerate(size_list):
        pred_dir = os.path.join(pred_root_dir, 'noaug_nodrop_'+losstype+'_iouthsp'+str(iou_ths)+'_dilation_'+str(idilation)+'_sz'+cur_sz+'_ds0_dc2'+'_wtta', 'rnd_0', 'pred_res')
        dmat = scio.loadmat(os.path.join(pred_dir, str(i)+'.mat'))
        s = dmat['instance'].astype(np.uint16)
        s_mask = np.logical_and(spred==0, s>0)
        max_intance_id = spred.max()
        s[s_mask] += max_intance_id
        spred[s_mask] = s[s_mask]
        #spred = s
    spred = remap_label(spred)
    gt = remap_label(gt)
    aji = get_fast_aji(gt, spred)
    print(i+1, aji)
    ajis.append(aji)
    if if_save:
        save_dir = os.path.join(pred_root_dir, 'final_res')
        confirm_loc(save_dir)
        scio.savemat(os.path.join(save_dir, str(i)+'.mat'), {'instance': spred, 'gt':gt})
avg_aji = np.mean(ajis)
print(avg_aji)