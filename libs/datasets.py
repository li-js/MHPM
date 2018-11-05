import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import time
import pickle

def get_resize_scale(im, MIN, MAX):
    im_h = im.shape[0]
    im_w = im.shape[1]
    im_min =  np.min((im_h, im_w))
    im_max =  np.max((im_h, im_w))
    scale = MIN/im_min
    if im_max*scale>MAX:
        scale = MAX/im_max
    return scale


class MHPGlobalDataSet(data.Dataset):
    def __init__(self, dat_list,  pre_scale=None, mean=(128, 128, 128), with_tag=False, tag_scaling=None,
                    crop_size=(321, 321), scale=True, mirror=False, ignore_label=255):
        self.dat_list = dat_list
        if crop_size is None:
            self.use_crop = False
        else:
            self.use_crop = True
            self.crop_h, self.crop_w = crop_size
        self.with_tag = with_tag
        self.tag_scaling =  tag_scaling  #If given, it should be a callable to get output size (about 1/8 of input)
        self.scale = scale # If true, randomly scale the image before cropping
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        if pre_scale is not None:
            self.pre_scale=True
            self.pre_scale_min = float(min(pre_scale))
            self.pre_scale_max = float(max(pre_scale))
        else:
            self.pre_scale=False
        assert(self.is_mirror is False), "Not implemented"

    def __len__(self):
        return len(self.dat_list)

    def generate_scale_label(self, image, label, label_tag=None):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        if label_tag is not None:
            label_tag = cv2.resize(label_tag, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
            return image, label, label_tag
        else:
            return image, label

    def __getitem__(self, index):
        dat = self.dat_list[index]
        image = cv2.imread(dat['filepath'], cv2.IMREAD_COLOR)
        label = cv2.imread(dat['global_mask_add'], cv2.IMREAD_GRAYSCALE)
        if self.with_tag:
            label_tag = cv2.imread(dat['global_tag_add'], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        #name = datafiles["name"]
        if self.pre_scale:
            pre_scale = get_resize_scale(image, self.pre_scale_min, self.pre_scale_max)
            image = cv2.resize(image, None, fx=pre_scale, fy=pre_scale, interpolation = cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=pre_scale, fy=pre_scale, interpolation = cv2.INTER_NEAREST)
            if self.with_tag:
                label_tag = cv2.resize(label_tag, None, fx=pre_scale, fy=pre_scale, interpolation = cv2.INTER_NEAREST)

        if self.use_crop is False:
            image = np.asarray(image, np.float32)
            image -= self.mean
            image = image.transpose((2, 0, 1))
            final_dat = {'image':image.copy(), 'label':label.copy(), 'filepath':dat['filepath']}
            if self.with_tag:
                if self.tag_scaling is None:
                    final_dat['label_tag']=label_tag.copy()
                else:
                    final_dat['label_tag']=cv2.resize(label_tag, (self.tag_scaling(label_tag.shape[1]), self.tag_scaling(label_tag.shape[0])), interpolation = cv2.INTER_NEAREST)
            return final_dat

        if self.scale:
            if self.with_tag:
                image, label, label_tag = self.generate_scale_label(image, label, label_tag=label_tag)
            else:
                image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                                            pad_w, cv2.BORDER_CONSTANT, 
                                            value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                                                pad_w, cv2.BORDER_CONSTANT,
                                                value=(self.ignore_label,))
            if self.with_tag:
                label_tag = cv2.copyMakeBorder(label_tag, 0, pad_h, 0, 
                                                pad_w, cv2.BORDER_CONSTANT,
                                                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        if self.with_tag:
            label_tag = np.asarray(label_tag[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        final_dat = {'image':image.copy(), 'label':label.copy(), 'filepath':dat['filepath']} 
        if self.with_tag:
            if self.tag_scaling is None:
                final_dat['label_tag'] = label_tag.copy()
            else:
                final_dat['label_tag'] = cv2.resize(label_tag, (self.tag_scaling(label_tag.shape[1]), self.tag_scaling(label_tag.shape[0])), interpolation = cv2.INTER_NEAREST)
        return final_dat


class MHPGlobalTestDataSet(data.Dataset):
    def __init__(self, dat_list, target_size=(600, 1000),mean=(128, 128, 128), ignore_label=255):        
        self.dat_list = dat_list
        self.mean = mean
        self.target_size_min = float(min(target_size))
        self.target_size_max = float(max(target_size))
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __len__(self):
        return len(self.dat_list)

    def get_resize_scale(self, im, MIN, MAX):
        im_h = im.shape[0]
        im_w = im.shape[1]
        im_min =  np.min((im_h, im_w))
        im_max =  np.max((im_h, im_w))
        scale = MIN/im_min
        if im_max*scale>MAX:
            scale = MAX/im_max
        return scale

    def __getitem__(self, index):
        dat = self.dat_list[index]
        image = cv2.imread(dat['filepath'], cv2.IMREAD_COLOR)
        label = cv2.imread(dat['global_mask_add'], cv2.IMREAD_GRAYSCALE)

        scale = self.get_resize_scale(image, self.target_size_min, self.target_size_max)
        image = cv2.resize(image, None, fx=scale, fy=scale)

        image = np.asarray(image, np.float32)

        image -= self.mean
        
        image = image.transpose((2, 0, 1))
        return {'image':image.copy(), 'label':label.copy(), 'filepath':dat['filepath']} 