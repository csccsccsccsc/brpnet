# brpnet
Pytorch Implement of Boundary-assisted Region Proposal Networks for Nucleus Segmentation (MICCAI 2020)
Paper can be found here: https://arxiv.org/abs/2006.02695.

Codes are built based on Python 3.6+ and Pytorch 1.1. 
For some reasons, there also exists MATLAB scripts in this project. The MATLAB scripts are tested based on MATLAB2016.

## Dataset:
We wrap all images into a .npy file (data_after_stain_norm_ref1.npy). Stain normalization is also performed.
Ground truth of segmentation and boundary is saved in anoth .npy file (gt.npy, bnd.npy).

The wrapped data file can be generated using the scripts in 'data_generator_scripts'.

## tafe
TAFE is the first stage of BRP-Net. 

### TAFE can be trained through:
python main_train_kfold_tafe.py 
(dataset loc and other settings can be modified in main_train_kfold_tafe.py)

### After training, evaluation is performed on validation set:
1) pred.py: predict resulst of validation set for all saved snapshots.
2) evaluate.py: evaluation for all saved snapshots.
3) find the best snapshots, and modify the settings in pred_testset.py and evaluate_testset.py
4) pred_testset.py
5) evaluate_testset.py

### prepare patch dataset for training patch-net
1) prepare_pred_trainset.py 
2) prepare_postproc_trainset.py
3) (MATLAB) prepare_match_instance_predictions.m
4) (MATLAB) prepare_save_matched_patches.m
(In prepare_save_matched_patches.m, training patch dataset and validation patch dataset are split.)

## patch-net
Patch-Net is the second stage of BRP-Net

### Train:
python main.py

### Evaluation:
1) python eval.py
2) python evaluate.py
3) python find_best_snapshot.py
4) python eval_testset.py
5) python evaluate_testset.py

### Contact:
mailbox: c.shengcong@mail.scut.edu.cn
