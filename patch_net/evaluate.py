import multiprocessing
import numpy as np
import scipy.io as scio

from metrics import get_fast_aji, remap_label

ndata = 16
ntrain = 4

max_epoch = 600
save_step = 20

net_val_dir = '/data0/cong/workplace/kumar_5fold/val/baseline_4fold_epoch600'
gts = np.load('/home/cong/workplace/kumar/gt.npy').astype(np.uint16)

list_dmat = scio.loadmat('./train/patchnet_trainset_list.mat');
valset = list_dmat['valset']

ths = 5
dilation_rate = 2

dilation_s = 0
dilation_c = 2
irnd = 0

ajis_allcp = []
for i_cp in range(save_step, max_epoch+1, save_step):
    ajis = []
    for i in valset:
        lbl = None
        for cursz in [48, 176]:
            dir_savefile = './val/noaug_nodrop_bceloss_iouthsp'+'ths'+str(ths)+'_dilation_'+str(dilation_rate)+'_sz'+str(cursz)+'_ds'+str(dilation_s)+'_dc'+str(dilation_c)+'_wtta/rnd_'+str(irnd)+'/'
            dmat = scio.loadmat(os.path.join(dir_savefile, 'snapshot_'+str(i_cp), str(i)+'.mat'))
            if lbl is None:
                lbl = dmat['instance']
            else:
                lbl_tmp = lbl.copy()
                max_num = instance_tmp.max()
                lbl[instance_tmp==0] = dmat['instance'][lbl_tmp==0] + max_num

        lbl = remap_label(lbl.astype(np.uint16))
        aji = get_fast_aji(remap_label(gts[i]), remap_label(lbl))
        #print(i, aji)
        ajis.append(aji)
    avg_aji = np.array(ajis).mean()
    ajis_allcp.append(avg_aji)
    print('Itrain {0:d}: Validation AJI of snapshot {1:d}: {2:.3f}'.format(itrain, i_cp, avg_aji))
maxaji_allcp = np.array(ajis_allcp).max()
icp = np.array(ajis_allcp).argmax()
np.save(os.path.join(pred_dir, 'val_aji.npy'), np.array(ajis_allcp))
print('Itrain {0:d}: the max validation AJI {1:.3f} in {2:d} snapshot'.format(itrain, maxaji_allcp, icp))