import sys
import argparse
import torch   # torch.__version__: '0.3.1'
import torch.nn as nn 
import numpy as np # np.__version__: '1.8.2'
import pickle, gzip
import cv2         # cv2.__version__: '2.4.8'
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
import random
import pickle

sys.path.append('libs/')
from model_solver import Res_Deeplab
import clustering

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 1
IGNORE_LABEL = 255
INPUT_SIZE = '600,1000'
NUM_CLASSES = 19
NUM_TAGS = 8


h, w = map(int, INPUT_SIZE.split(','))
input_size = (h, w)

cudnn.enabled = False
gpu = 0

# Create network.
model = Res_Deeplab(num_classes=NUM_CLASSES, num_tags=NUM_TAGS, split_layer=4)

saved_state_dict = torch.load('models/models-MHPM-SD2_plus.pth', map_location=torch.device('cpu') )
model.load_state_dict(saved_state_dict)

model.eval() 
# model.cuda(0)
# cudnn.benchmark = False

im_add = 'im_test.jpg'
im = cv2.imread(im_add, cv2.IMREAD_COLOR)
im_h = im.shape[0]
im_w = im.shape[1]
im_min =  np.min((im_h, im_w))
im_max =  np.max((im_h, im_w))
scale = 1.0*input_size[0]/im_min
if im_max*scale>input_size[1]:
    scale = 1.0*input_size[1]/im_max

im = cv2.resize(im, None, fx=scale, fy=scale)
im = np.asarray(im, np.float32)
im -= IMG_MEAN

im = Variable(torch.tensor(im.transpose((2, 0, 1))[np.newaxis]))
model.cpu()
model.eval()
out_seg, out_tag = model(im)

sm = torch.nn.Softmax2d()
interp = torch.nn.Upsample(size=(im_h, im_w), mode='bilinear')
seg_prob = interp(sm(out_seg)).data[0].cpu()
out_tag  = interp(out_tag).data[0].cpu()


_, seg_cls = torch.max(seg_prob, 0, keepdim=True)
instance_map_torch = clustering.clustering(out_tag, seg_cls, 1.5, perform_opening=True)

seg_prob = seg_prob.numpy().transpose((1,2,0))
mask_global = seg_prob.argmax(axis=2)


from pylab import plt
plt.figure('im')
plt.imshow(cv2.imread(im_add, cv2.IMREAD_COLOR)[:,:,::-1])
plt.figure('Segmentation result')
plt.imshow(mask_global)
plt.figure('instance map')
plt.imshow(instance_map_torch.numpy())
plt.show()
