import os
import numpy as np
#import scipy.io as scio
import torch
#import random
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.interpolation import map_coordinates
from PIL import Image, ImageEnhance

from custom import confirm_loc

def elastic_transform(shape, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    # This function get indices only.
    if random_state is None:
        random_state = np.random.RandomState(None)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    return indices

# Image Net mean and std
norm_mean=[0.485, 0.456, 0.406]
norm_std=[0.229, 0.224, 0.225]

class KumarDataset(torch.utils.data.Dataset):
    def __init__(self, loc_head, list, crop_size=[256, 256]):
        self.loc_head = loc_head

        self.imgs = np.load(os.path.join(loc_head, 'data_after_stain_norm_ref1.npy'))
        self.imgs = self.imgs[list]

        self.seg_labels = np.load(os.path.join(loc_head, 'gt.npy'))
        self.seg_labels = self.seg_labels[list]
        # Instance labels to segmentation labels
        self.seg_labels[self.seg_labels>0] = 1

        self.bnd_labels = np.load(os.path.join(loc_head, 'bnd.npy'))
        self.bnd_labels = self.bnd_labels[list]
        
        self.naug = 6#ori*1 + rotate*3 + flip*2
        self.nimg = len(self.imgs)
        self.crop_size = crop_size

    def __len__(self):
        return self.nimg*self.naug

    def __getitem__(self, idx):
    
        iaug = int(np.mod(idx, self.naug))
        index = int(np.floor(idx/self.naug))
        
        # While doing random augmentation here (instead of calling transformer) with multi-workers, all the workers get the same numpy random state sometimes. To avoid this, call np.random.seed() again here.
        np.random.seed()
        #print(idx, np.random.rand(10))

        img = self.imgs[index].copy()
        h, w, mod = img.shape

        # Color Jittering
        # Color, Brightness, Contrast, Sharpness
        rnd_factor = np.random.rand()*0.1+0.9
        img = Image.fromarray(img.astype(np.uint8))
        img = ImageEnhance.Color(img).enhance(rnd_factor)
        rnd_factor = np.random.rand()*0.1+0.9
        img = ImageEnhance.Brightness(img).enhance(rnd_factor)
        rnd_factor = np.random.rand()*0.1+0.9
        img = ImageEnhance.Contrast(img).enhance(rnd_factor)
        rnd_factor = np.random.rand()*0.2+0.9
        img = ImageEnhance.Sharpness(img).enhance(rnd_factor)
        img = np.asarray(img).astype(np.float32)
        img = img.transpose([2, 0, 1])
        for imod in list(range(mod)):
            img[imod] = (img[imod]/255.0 - norm_mean[imod])/norm_std[imod]
        img += np.random.normal(0, np.random.rand(), img.shape)*0.01

        seg_label = self.seg_labels[index].copy()
        bnd_label = self.bnd_labels[index].copy()

        # Crop
        sh = np.random.randint(0, h-self.crop_size[0]-1)
        sw = np.random.randint(0, w-self.crop_size[1]-1)
        img = img[:, sh:(sh+self.crop_size[0]), sw:(sw+self.crop_size[1])]
        seg_label = seg_label[sh:(sh+self.crop_size[0]), sw:(sw+self.crop_size[1])]
        bnd_label = bnd_label[sh:(sh+self.crop_size[0]), sw:(sw+self.crop_size[1])]

        # assert(iaug>0 and iaug<6)
        # Aug
        if iaug<=3 and iaug>0:
            img = np.rot90(img, iaug, axes=(len(img.shape)-2, len(img.shape)-1))
            seg_label = np.rot90(seg_label, iaug, axes=(len(seg_label.shape)-2, len(seg_label.shape)-1))
            bnd_label = np.rot90(bnd_label, iaug, axes=(len(bnd_label.shape)-2, len(bnd_label.shape)-1))
        elif iaug>=4:
            img = np.flip(img, len(img.shape)-(iaug-3))
            seg_label = np.flip(seg_label, len(seg_label.shape)-(iaug-3))
            bnd_label = np.flip(bnd_label, len(bnd_label.shape)-(iaug-3))

        if np.random.rand()>=0.5:
            rnd_et = np.random.rand(2)
            indices = elastic_transform(seg_label.shape, int(rnd_et[0]*20), 5*(rnd_et[1]+1.0))
            for imod in range(mod):
                img[imod] = map_coordinates(img[imod].squeeze(), indices, order=1, mode='reflect').reshape(img[imod].shape)
            seg_label = map_coordinates(seg_label.squeeze(), indices, order=1, mode='reflect').reshape(seg_label.shape)
            bnd_label = map_coordinates(bnd_label.squeeze(), indices, order=1, mode='reflect').reshape(bnd_label.shape)

        img = img.copy()
        seg_label = seg_label.copy()
        bnd_label = bnd_label.copy()
        
        return img, seg_label, bnd_label