from DenseEncoderIAMUShapeDecoder import * 
import numpy as np
import os
import time
import scipy.io as scio
import random
from adamw_r.adamw import AdamW
from adamw_r.cyclic_scheduler import CyclicLRWithRestarts, ReduceMaxLROnRestart

from dataset import KumarDataset
from custom import *
from loss import dice_loss, smooth_truncated_loss, compute_loss_list

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.set_num_threads(8)

batch_size = 3
val_batch_size = 2
MAX_epoch = 600
save_cp_after_n_epoch = 20

loaded_epoch = -1
n_pred_labels_type = 1
n_pred_labels_bnd = 1

input_modalities = 3
weight_decay = 1e-5
lr = 3e-4
tuning_epoch = 0
increasement = lr/max(tuning_epoch, 1)
dice_weight = 0.5
lr_redecay = ReduceMaxLROnRestart(0.5)
crop_size = 256

train_data_loc = '/home/cong/workplace/kumar/'

ndata = 16#len(os.listdir(os.path.join(train_data_loc, 'Images')))
ntrain = 4
kfold_train_idx, kfold_val_idx = kfold_list(ndata, ntrain)

print('Train Lists : ', kfold_train_idx)
print('Val Lists : ', kfold_val_idx)

for iseed in list(range(ntrain)):
    dir_checkpoint = './weight/baseline_'+str(ntrain)+'fold_epoch'+str(MAX_epoch)+'/rnd_'+str(iseed)+'/'
    confirm_loc(dir_checkpoint)

    net = UNet(input_modalities, n_pred_labels_type, n_pred_labels_bnd).cuda()

    train_data = KumarDataset(loc_head=train_data_loc, list=kfold_train_idx[iseed], crop_size=[crop_size, crop_size])
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)

    optimizer = AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CyclicLRWithRestarts(optimizer, batch_size, len(train_data), restart_period=50, t_mult=2.0, policy="cosine", eta_on_restart_cb=lr_redecay)
    
    net.train()

    slosses = []
    closses = []
    dslosses = []
    dclosses = []

    for i in list(range(tuning_epoch)):
        replace_learning_rate(optimizer, increasement*(i+1))
        print('->', str(i)+'th warming up epoch, '+'cur_lr '+str(optimizer.param_groups[0]['lr']))
        for i_data, (image, label, boundary) in enumerate(train_dataloader):
            st = time.time()
            #net.train()
            optimizer.zero_grad()
            b, _, h, w = image.shape

            image = image.float().cuda()

            label = label.float().cuda().view(b, 1, h, w)
            boundary = boundary.float().cuda().view(b, 1, h, w)
            label_s1 = F.interpolate(label, scale_factor=0.5, mode='bilinear')
            label_s2 = F.interpolate(label, scale_factor=0.25, mode='bilinear')
            label_s3 = F.interpolate(label, scale_factor=0.125, mode='bilinear')
            boundary_s1 = F.interpolate(boundary, scale_factor=0.5, mode='bilinear')
            boundary_s2 = F.interpolate(boundary, scale_factor=0.25, mode='bilinear')
            boundary_s3 = F.interpolate(boundary, scale_factor=0.125, mode='bilinear')


            sout, sout_0, sout_1, sout_2, sout_3, cout, cout_0, cout_1, cout_2, cout_3 = net(image)
            
            seg_stl_losses = compute_loss_list(smooth_truncated_loss, [sout, sout_0, sout_1, sout_2, sout_3], [label, label, label_s1, label_s2, label_s3])
            seg_dsc_losses = compute_loss_list(dice_loss, [sout], [label])

            bnd_stl_losses = compute_loss_list(smooth_truncated_loss, [cout, cout_0, cout_1, cout_2, cout_3], [boundary, boundary, boundary_s1, boundary_s2, boundary_s3])
            bnd_dsc_losses = compute_loss_list(dice_loss, [cout], [boundary])

            loss = 0.0
            for iloss in seg_stl_losses:
                loss += iloss
            for iloss in seg_dsc_losses:
                loss += iloss*dice_weight
            for iloss in bnd_stl_losses:
                loss += iloss
            for iloss in bnd_dsc_losses:
                loss += iloss*dice_weight

            loss.backward()
            optimizer.step()

            et = time.time()
            print('warmming up epoch {0: d}, batch {1:d} ; sloss {3:.3f}, {4:.3f}; closs {5:.3f}, {6:.3f}; used {2:.6f} s'\
                    .format(i, i_data, et-st, \
                    seg_stl_losses[0], seg_dsc_losses[0], bnd_stl_losses[0], bnd_dsc_losses[0]))

    for i in list(range(MAX_epoch)):

        scheduler.step()

        slosses_insideepoch = []
        closses_insideepoch = []
        dslosses_insideepoch = []
        dclosses_insideepoch = []

        net.train(True)

        print('->', str(i)+'th training epoch, '+'cur_lr '+str(optimizer.param_groups[0]['lr']))

        for i_data, (image, label, boundary) in enumerate(train_dataloader):
        
            st = time.time()
            #net.train()
            optimizer.zero_grad()
            b, _, h, w = image.shape

            image = image.float().cuda()
            label = label.float().cuda().view(b, 1, h, w)
            boundary = boundary.float().cuda().view(b, 1, h, w)
            label_s1 = F.interpolate(label, scale_factor=0.5, mode='bilinear')
            label_s2 = F.interpolate(label, scale_factor=0.25, mode='bilinear')
            label_s3 = F.interpolate(label, scale_factor=0.125, mode='bilinear')
            boundary_s1 = F.interpolate(boundary, scale_factor=0.5, mode='bilinear')
            boundary_s2 = F.interpolate(boundary, scale_factor=0.25, mode='bilinear')
            boundary_s3 = F.interpolate(boundary, scale_factor=0.125, mode='bilinear')

            b, _, h, w = image.shape

            sout, sout_0, sout_1, sout_2, sout_3, cout, cout_0, cout_1, cout_2, cout_3 = net(image)
            
            seg_stl_losses = compute_loss_list(smooth_truncated_loss, [sout, sout_0, sout_1, sout_2, sout_3], [label, label, label_s1, label_s2, label_s3])
            seg_dsc_losses = compute_loss_list(dice_loss, [sout], [label])

            bnd_stl_losses = compute_loss_list(smooth_truncated_loss, [cout, cout_0, cout_1, cout_2, cout_3], [boundary, boundary, boundary_s1, boundary_s2, boundary_s3])
            bnd_dsc_losses = compute_loss_list(dice_loss, [cout], [boundary])

            loss = 0.0
            for iloss in seg_stl_losses:
                loss += iloss
            for iloss in seg_dsc_losses:
                loss += iloss*dice_weight
            for iloss in bnd_stl_losses:
                loss += iloss
            for iloss in bnd_dsc_losses:
                loss += iloss*dice_weight

            slosses_insideepoch.append(seg_stl_losses[0].item())
            closses_insideepoch.append(bnd_stl_losses[0].item())
            dslosses_insideepoch.append(seg_dsc_losses[0].item())
            dclosses_insideepoch.append(bnd_dsc_losses[0].item())

            loss.backward()
            optimizer.step()
            scheduler.batch_step()

            et = time.time()

            print('epoch {0: d}, batch {1:d} ; sloss {3:.3f}, {4:.3f}; closs {5:.3f}, {6:.3f}; used {2:.6f} s'\
                    .format(i, i_data, et-st, \
                    seg_stl_losses[0], seg_dsc_losses[0], bnd_stl_losses[0], bnd_dsc_losses[0]))

        meansloss_epoch = np.array(slosses_insideepoch).mean()
        slosses.append(meansloss_epoch)
        meancloss_epoch = np.array(closses_insideepoch).mean()
        closses.append(meancloss_epoch)
        meandsloss_epoch = np.array(dslosses_insideepoch).mean()
        dslosses.append(meandsloss_epoch)
        meandcloss_epoch = np.array(dclosses_insideepoch).mean()
        dclosses.append(meandcloss_epoch)
        if np.mod(i+1, save_cp_after_n_epoch)==0 and i>0:
            torch.save(net.state_dict(), dir_checkpoint + 'model_of_' + str(i+1) + '.pth')
            #torch.save(optimizer.state_dict(), dir_checkpoint + 'optim_of_' + str(i+1) + '.pth')
            scio.savemat(dir_checkpoint+'losses.mat', {'s':np.array(slosses), 'c':np.array(closses), 'ds':np.array(dslosses), 'dc':np.array(dclosses)})
            