import os
import numpy as np
import h5py
import torch
import scipy.io as scio
from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter, median_filter
import random
import copy
from PIL import Image, ImageEnhance

def confirm_loc(loc):
    isExists=os.path.exists(loc)
    if not isExists:
        os.makedirs(loc)

def elastic_transform(shape, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    return indices

norm_mean=[0.485, 0.456, 0.406]
norm_std=[0.229, 0.224, 0.225]

class PatchDataset(object):
    def __init__(self, loc_head, matname, sz, dilation_s=0, dilation_c=0):
        self.loc_head = loc_head

        h5fileobj = h5py.File(loc_head +matname+'.mat', 'r')
        print(loc_head +matname+str(sz)+'.mat')

        # Note! [mod, w, h, N] or [w, h, N]
        self.data = h5fileobj['resz_imgs'][:].astype(np.uint8)
        self.gts = h5fileobj['resz_gts'][:].astype(np.float32)
        self.spreds = h5fileobj['resz_orispreds'][:].astype(np.float32)
        self.cpreds = h5fileobj['resz_oricpreds'][:].astype(np.float32)
        self.oripreds = h5fileobj['resz_oripreds'][:].astype(np.float32)
        
        self.data = self.data.transpose([3, 2, 1, 0])
        self.gts = self.gts.transpose([2, 1, 0])

        self.spreds = self.spreds.transpose([2, 1, 0])
        self.cpreds = self.cpreds.transpose([2, 1, 0])
        self.oripreds = self.oripreds.transpose([2, 1, 0])
        h5fileobj.close()

        # # If the data are saved using v7.3 matlab save function, h5py should be used to load the .mat file; if not, scio.loadmat can be used.
        # dmat = scio.loadmat(loc_head +matname+str(sz)+'.mat')
        # self.data = dmat['resz_imgs'].astype(np.uint8)
        # self.gts = dmat['resz_gts'].astype(np.float32)
        # self.spreds = dmat['resz_orispreds'].astype(np.float32)
        # self.cpreds = dmat['resz_oricpreds'].astype(np.float32)
        # self.oripreds = dmat['resz_oripreds'].astype(np.float32)
        print(self.data.shape, self.gts.shape, self.spreds.shape, self.cpreds.shape, self.oripreds.shape)

        self.sz = sz
        
        self.ndata = self.data.shape[0]
        # dilation:
        # 0: instnace mask * prob
        # >0: dilated instance mask * prob
        # <0: original prob(with all instances within the patch)
        self.dilation_s = dilation_s
        self.dilation_c = dilation_c

    def __len__(self):
        return self.ndata

    def __getitem__(self, index):
        np.random.seed()

        h, w, mod = self.data[index].shape

        dimg = Image.fromarray(self.data[index])
        l = self.gts[index].copy()
        sin = self.spreds[index].copy()
        cin = self.cpreds[index].copy()
        
        oripredmask_ori = self.oripreds[index].copy()
        
        if self.dilation_s > 0:
            sin *= ndimage.morphology.binary_dilation(oripredmask_ori>=0.5, iterations=self.dilation_s).astype(np.float32)
        elif self.dilation_s == 0:
            sin *= oripredmask_ori.astype(np.float32)

        if self.dilation_c > 0:
            cin *= ndimage.morphology.binary_dilation(oripredmask_ori>=0.5, iterations=self.dilation_c).astype(np.float32)
        elif self.dilation_c == 0:
            cin *= oripredmask_ori.astype(np.float32)

        #Color Jittering
        #Color, Brightness, Contrast, Sharpness
        #0: black-white image -> 1:ori image
        rnd_factor = np.random.rand()*0.1+0.9
        dimg = ImageEnhance.Color(dimg).enhance(rnd_factor)

        #0: black image -> 1:ori image
        rnd_factor = np.random.rand()*0.1+0.9
        dimg = ImageEnhance.Brightness(dimg).enhance(rnd_factor)
        
        #0: solid grey image -> 1:ori image
        rnd_factor = np.random.rand()*0.1+0.9
        dimg = ImageEnhance.Contrast(dimg).enhance(rnd_factor)
        
        #0: blurred image -> 1:ori image -> 2:sharp image
        rnd_factor = np.random.rand()*0.2+0.9
        dimg = ImageEnhance.Sharpness(dimg).enhance(rnd_factor)
        
        d = np.asarray(dimg).astype(np.float32)

        d = d.transpose([2, 0, 1])
        for imod in list(range(mod)):
            d[imod] = (d[imod]/255.0 - norm_mean[imod])/norm_std[imod]

        #elastic_transform
        if np.random.rand()>=0.5:
            rnd_et = np.random.rand(2)
            indices = elastic_transform(sin.shape, int(rnd_et[0]*self.sz*0.5), 5*(rnd_et[1]+1.0))
            sin = map_coordinates(sin.squeeze(), indices, order=1, mode='reflect').reshape(sin.shape)
            cin = map_coordinates(cin.squeeze(), indices, order=1, mode='reflect').reshape(cin.shape)

        # Gaussian OR Median Blurred
        rnd_blur = np.random.rand(mod+1)
        if rnd_blur[0]>=0.667:
            for imod in list(range(mod)):
                d[imod] = gaussian_filter(d[imod], sigma=rnd_blur[imod])
            sin = gaussian_filter(sin, sigma=np.random.rand())
            cin = gaussian_filter(cin, sigma=np.random.rand())
        elif rnd_blur[0]>=0.333:
            for imod in list(range(mod)):
                d[imod] = median_filter(d[imod], size=int(2*rnd_blur[imod])*2+1)
            sin = median_filter(sin, size=int(2*rnd_blur[imod])*2+1)
            cin = median_filter(cin, size=int(2*rnd_blur[imod])*2+1)
        d2 = d.copy()
        l2 = l.copy()
        sin2 = sin.copy()
        cin2 = cin.copy()

        return d2, l2, sin2, cin2