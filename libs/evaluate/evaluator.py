import os
import numpy as np
from PIL import Image

class ParsingEval:
    def __init__(self, nb_class=19, model_add=''):
        self.model_add=model_add
        self.NUM_CLS=nb_class
        self.hist=np.zeros((self.NUM_CLS, self.NUM_CLS))

        self.classes =[
        "background", 
        "hat", 
        "hair", 
        "sunglass", 
        "upper-clothes", 
        "skirt", 
        "pants", 
        "dress", 
        "belt", 
        "left-shoe", 
        "right-shoe", 
        "face", 
        "left-leg", 
        "right-leg", 
        "left-arm", 
        "right-arm", 
        "bag", 
        "scarf", 
        "torso-skin" ]


    def fast_hist(self, a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

    def update(self, image_array, label_array):     
        #label_array = np.array(Image.open(label_path), dtype=np.int32)
        #image_array = np.array(Image.open(img_path), dtype=np.int32)
        
        gtsz = label_array.shape
        imgsz = image_array.shape
        #if not gtsz == imgsz:
        #   image = image.resize((gtsz[1], gtsz[0]), Image.ANTIALIAS)
        #   image_array = np.array(image, dtype=np.int32)
        assert(np.allclose(gtsz, imgsz)), 'Image and Label should have the same size'
        self.hist += self.fast_hist(label_array, image_array, self.NUM_CLS)

    def show_result(self):
        hist=self.hist
        # num of correct pixels
        num_cor_pix = np.diag(hist)
        # num of gt pixels
        num_gt_pix = hist.sum(1)
        print '=' * 50

        # @evaluation 1: overall accuracy
        acc = num_cor_pix.sum() / hist.sum()
        print '>>>', 'overall accuracy', acc
        print '-' * 50

        # @evaluation 2: mean accuracy & per-class accuracy 
        print 'Accuracy for each class (pixel accuracy):'
        for i in xrange(self.NUM_CLS):
            print('%-15s: %f' % (self.classes[i], num_cor_pix[i] / num_gt_pix[i])) 
        acc = num_cor_pix / num_gt_pix
        print '>>>', 'mean accuracy', np.nanmean(acc)
        print '-' * 50
        
        # @evaluation 3: mean IU & per-class IU
        union = num_gt_pix + hist.sum(0) - num_cor_pix
        for i in xrange(self.NUM_CLS):
            print('%-15s: %f' % (self.classes[i], num_cor_pix[i] / union[i]))
        iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
        print '>>>', 'mean IU', np.nanmean(iu)
        print '-' * 50

        # @evaluation 4: frequency weighted IU
        freq = num_gt_pix / hist.sum()
        print '>>>', 'fwavacc', (freq[freq > 0] * iu[freq > 0]).sum()
        print '=' * 50

    def get_mean_iou(self):
        hist=self.hist
        num_cor_pix = np.diag(hist)
        num_gt_pix = hist.sum(1)
        union = num_gt_pix + hist.sum(0) - num_cor_pix
        iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
        return np.nanmean(iu)

    def get_all_iou(self):
        hist=self.hist
        num_cor_pix = np.diag(hist)
        num_gt_pix = hist.sum(1)
        union = num_gt_pix + hist.sum(0) - num_cor_pix
        iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
        return iu

    def get_acc(self):
        hist=self.hist
        num_cor_pix = np.diag(hist)
        acc = num_cor_pix.sum() / hist.sum()
        return acc


    def cal_one_mean_iou(self, image_array, label_array):
        hist = self.fast_hist(label_array, image_array, self.NUM_CLS).astype(np.float)
        num_cor_pix = np.diag(hist)
        num_gt_pix = hist.sum(1)
        union = num_gt_pix + hist.sum(0) - num_cor_pix
        iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
        return iu
        mean_iou = np.nanmean(iu)        

        #freq = num_gt_pix / hist.sum()
        #fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()
        
