.m scipts under this filefold are used to generate the wrapped data in brp-net.

Take Kumar dataset as an example:
1) After downloading the dataset and modifying the dataset path and the save path in transform_dataset_anno.m, run transform_dataset_anno.m
2) run seg_and_bnd.m and save_stainnorm_mm.m
3) The generated .mat data needs to be further transform to .npy data in python.
