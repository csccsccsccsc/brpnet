import multiprocessing
import numpy as np
import scipy.io as scio
from skimage import io
from metrics import get_fast_aji, remap_label
from post_proc import post_proc
from custom import *
import multiprocessing

# snapshot_lists: Select From Trained Models
configs = {
    'ndata': 14,
    'ntrain': 4,
    'save_cp_after_n_epoch': 20,
    'net_val_dir': './test/tafe_4fold_epoch600',
    'save_dir': './test/tafe_4fold_epoch600',
    'snapshot_lists': [[27], [4], [9], [8]],
    'nprocesses': 4
}




def processing_func(i, configs):
    ndata = configs['ndata']
    ntrain = configs['ntrain']
    save_cp_after_n_epoch = configs['save_cp_after_n_epoch']
    net_val_dir = configs['net_val_dir']
    save_dir = configs['save_dir']
    snapshot_lists = configs['snapshot_lists']

    spred = None
    cpred = None
    count = None
    for itrain in range(ntrain):
        pred_dir = os.path.join(net_val_dir, 'rnd_'+str(itrain))
        print(pred_dir)
        for i_cp in snapshot_lists[itrain]:
            dmat = scio.loadmat(os.path.join(pred_dir, 'snapshot_'+str(i_cp*save_cp_after_n_epoch), str(i)+'.mat'))
            s = dmat['s']
            c = dmat['c']
            if len(s.shape)>2:
                s = np.reshape(s, [s.shape[1], s.shape[2]])
            if len(c.shape)>2:
                c = np.reshape(c, [c.shape[1], c.shape[2]])
            if spred is not None:
                spred += s
                cpred += c
                count += 1.0
            else:
                spred = s
                cpred = c
                count = np.ones(s.shape, dtype=np.float32)
    spred /= count
    cpred /= count
    lbl_d0 = post_proc(spred-cpred, post_dilation_iter=0).astype(np.uint16)
    lbl_d2 = post_proc(spred-cpred, post_dilation_iter=2).astype(np.uint16)
    scio.savemat(os.path.join(save_dir, 'pred_res_'+str(i)+'_withpostproc_2.mat'), {'s':spred.astype(np.float32), 'c':cpred.astype(np.float32), 'instance':lbl_d2, 'instance_nodilation':lbl_d0})



ndata = configs['ndata']
p=multiprocessing.Pool(processes=configs[nprocesses])
for i in range(ndata):
    p.apply_async(processing_func, (i, configs))
p.close()
p.join()

