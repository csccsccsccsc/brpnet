import os
import numpy as np
import scipy.io as scio
from metrics import get_fast_aji, remap_label
import glob

gts = np.load('/home/cong/workplace/kumar/gt.npy').astype(np.uint16)

# find the best snapshot
for ifold in range(1,4):
    list_dmat = scio.loadmat('../tafe/train/tafe_4fold_epoch600/patchnet_trainset_list_ifold'+str(ifold)+'.mat');
    valset = list_dmat['valset']
    print(valset)
    valset -= 1
    for cur_sz in ['48', '176']:
        with open(os.path.join('./val/noaug_nodrop_focalloss_iouthsp5_dilation_2_sz'+str(cur_sz)+'_ds0_dc2_ifold'+str(ifold)+'_wtta/rnd_0', 'val_res.txt'), 'w') as f:
            cur_bset_aji = -1.0
            cur_best_aji_isnapshot = -1
            dir_checkpoint = './weight/noaug_nodrop_focalloss_iouthsp5_dilation_2_sz'+str(cur_sz)+'_ds0_dc2_ifold'+str(ifold)+'/rnd_0/'
            snapshot_list = glob.glob(dir_checkpoint+'*.pth')
            for isnapshot_name in snapshot_list:
                isnapshot = int(isnapshot_name.split('/')[-1].split('.')[0].split('_')[-1])
                # if isnapshot <= 10:
                #     continue
                val_pred_filefold = './val/noaug_nodrop_focalloss_iouthsp5_dilation_2_sz'+str(cur_sz)+'_ds0_dc2_ifold'+str(ifold)+'_wtta/rnd_0/snapshot_'+str(isnapshot)
                ajis = []
                for i_data in valset[0]:
                    pred_dmat = scio.loadmat(os.path.join(val_pred_filefold, str(i_data)+'.mat'))
                    pred_lbl_map = pred_dmat['instance'].copy()
                    if len(np.unique(pred_lbl_map[pred_lbl_map>0])) >= 1:
                        pred_lbl_map = remap_label(pred_lbl_map)
                    aji = get_fast_aji(remap_label(gts[i_data]), pred_lbl_map)
                    ajis.append(aji)
                avg_aji = np.mean(ajis)
                if avg_aji >= cur_bset_aji:
                    cur_bset_aji = avg_aji
                    cur_best_aji_isnapshot = isnapshot
                info2print = 'Size: {:s}, snapshot: {:d}, cur aji: {:.5f}, best aji: {:.5f} in isnapshot: {:d}'.format(cur_sz, isnapshot, avg_aji, cur_bset_aji, cur_best_aji_isnapshot)
                print(info2print)
                f.write(info2print+'\n')
