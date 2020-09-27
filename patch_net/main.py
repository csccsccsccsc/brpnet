import numpy as np
import os
import time
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import scipy.io as scio

from losses import *
from patch_dense_net import UNet
from adamw_r.adamw import AdamW
from adamw_r.cyclic_scheduler import CyclicLRWithRestarts, ReduceMaxLROnRestart
from dataset import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.set_num_threads(8)

reszs = [48, 176]
batch_sizes = [16, 4]
num_steps = [100, 200]
ths_list = [5] # ths*0.1
dilation_s = 0
dilation_c = 2
dilation_list = [2]
basic_dir_checkpoint = './weight/noaug_nodrop_focalloss_iouthsp'
data_loc = '../tafe/train/tafe_4fold_epoch600/predmatchgt/'
for ifold in range(4):
    for ths in ths_list:
        for dilation in dilation_list:
            for resz, batch_size, num_step in zip(reszs, batch_sizes, num_steps):
                for irnd in range(1):
                    MAX_epoch = 10
                    loaded_epoch = -1
                    n_pred_labels = 1
                    input_modalities = 3
                    weight_decay = 1e-5
                    lr = 3e-4
                    dice_weight = 0.5

                    dir_checkpoint = basic_dir_checkpoint+str(ths)+'_dilation_'+str(dilation)+'_sz'+str(resz)+'_ds'+str(dilation_s)+'_dc'+str(dilation_c)+'_ifold'+str(ifold)+'/rnd_'+str(irnd)+'/'
                    confirm_loc(dir_checkpoint)
                    
                    print(dir_checkpoint)

                    patch_dataset = PatchDataset(loc_head=data_loc, matname='reszdata_withmaskedsc_iouths'+str(ths)+'_idilate'+str(dilation)+'_ushapedecoder'+str(resz)+'_ifold'+str(ifold), \
                                                    sz=resz, dilation_s=dilation_s, dilation_c=dilation_c)

                    niter_perepoch = len(patch_dataset)//batch_size
                    nstep = np.ceil(MAX_epoch*niter_perepoch/num_step)
                    print(len(patch_dataset))
                    print(nstep)

                    net = UNet(input_modalities, n_pred_labels*2, n_pred_labels).cuda()
                    optimizer = AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nstep, eta_min=0, last_epoch=-1)
                    dataset = data.DataLoader(patch_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

                    slosses = []
                    cur_savesnapshot = 0
                    for i in list(range(MAX_epoch)):
                        slosses_insideepoch = []
                        net.train(True)

                        print('->', str(i)+'th training epoch, '+'cur_lr '+str(optimizer.param_groups[0]['lr']))

                        for i_data, (image, label, s_in, c_in) in enumerate(dataset):

                            st = time.time()

                            image = image.float().cuda()
                            label = label.float().unsqueeze(dim=1).cuda()
                            s_in = s_in.float().unsqueeze(dim=1).cuda()
                            c_in = c_in.float().unsqueeze(dim=1).cuda()

                            b, _, h, w = image.shape

                            sout = net(image, torch.cat((s_in, c_in), dim=1))

                            loss = focal_loss(sout, label) + dice_weight*dice_loss_perimg(F.sigmoid(sout), label)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            slosses_insideepoch.append(loss.item())
                            
                            et = time.time()

                            print('epoch {0: d}, batch {1:d} ; sloss {3:.3f}; used {2:.6f} s'\
                                    .format(i, i_data, et-st, loss))

                            if (i_data>=1 and np.mod(i_data+niter_perepoch*i, num_step)==0) or (i_data==len(patch_dataset)-1 and i==MAX_epoch-1):
                                torch.save(net.state_dict(), dir_checkpoint + 'model_of_' + str(cur_savesnapshot+1) + '.pth')
                                cur_savesnapshot+=1
                                scheduler.step()
                                print('Saving snapshot and start next step')

                        meansloss_epoch = np.array(slosses_insideepoch).mean()
                        slosses.append(meansloss_epoch)

                    scio.savemat(dir_checkpoint+'losses.mat', {'s':np.array(slosses)})