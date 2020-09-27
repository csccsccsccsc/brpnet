import multiprocessing
import numpy as np
import scipy.io as scio

from metrics import get_fast_aji, remap_label
from post_proc import post_proc
from custom import *

ndata = 14
ntrain = 4
save_cp_after_n_epoch = 20

net_val_dir = './test/tafe_4fold_epoch600'
gts = np.load('/home/cong/workplace/kumar/valgt.npy').astype(np.uint16)

# Select From Trained Models
# TAFE:
snapshot_lists = [[27], [4], [9], [8]]
# Baseline:
# snapshot_lists = [[10], [4], [20], [15]]

ajis = []
for i, gt in enumerate(gts):
    spred = np.zeros(gt.shape, dtype=np.float32)
    cpred = np.zeros(gt.shape, dtype=np.float32)
    count = np.zeros(gt.shape, dtype=np.float32)
    for itrain in range(ntrain):
        pred_dir = os.path.join(net_val_dir, 'rnd_'+str(itrain))
        for i_cp in snapshot_lists[itrain]:
            dmat = scio.loadmat(os.path.join(pred_dir, 'snapshot_'+str(i_cp*save_cp_after_n_epoch), str(i)+'.mat'))
            s = dmat['s']
            c = dmat['c']
            if len(s.shape)>2:
                s = np.reshape(s, [s.shape[1], s.shape[2]])
            if len(c.shape)>2:
                c = np.reshape(c, [c.shape[1], c.shape[2]])
            spred += s
            cpred += c
            count += 1.0
    spred /= count
    cpred /= count
    lbl = post_proc(spred-cpred).astype(np.uint16)
    aji = get_fast_aji(remap_label(gt), remap_label(lbl))
    # print(i, aji)
    # scio.savemat()
    ajis.append(aji)
avg_aji = np.array(ajis).mean()
print(avg_aji)