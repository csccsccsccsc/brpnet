import multiprocessing
import numpy as np
import scipy.io as scio

from metrics import get_fast_aji, remap_label
from post_proc import post_proc
from custom import *

ndata = 16
ntrain = 4
kfold_train_idx, kfold_val_idx = kfold_list(ndata, ntrain, seed=123)

max_epoch = 600
save_step = 20

net_val_dir = '/data0/cong/workplace/kumar_5fold/val/baseline_4fold_epoch600'
gts = np.load('/home/cong/workplace/kumar/gt.npy').astype(np.uint16)
for itrain in range(ntrain):
    ajis_allcp = []
    pred_dir = os.path.join(net_val_dir, 'rnd_'+str(itrain))
    for i_cp in range(save_step, max_epoch+1, save_step):
        ajis = []
        for i in kfold_val_idx[itrain]:
            dmat = scio.loadmat(os.path.join(pred_dir, 'snapshot_'+str(i_cp), str(i)+'.mat'))
            s = dmat['s']
            c = dmat['c']
            if len(s.shape)>2:
                s = np.reshape(s, [s.shape[1], s.shape[2]])
            if len(c.shape)>2:
                c = np.reshape(c, [c.shape[1], c.shape[2]])
            lbl = post_proc(s-c, post_dilation_iter=2).astype(np.uint16)
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