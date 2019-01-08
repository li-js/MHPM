import os
import cv2
from PIL import Image
import numpy as np
from pylab import plt
ann_dict={}
im_list=sorted(os.listdir('images/'))
for add in im_list:
    ann_dict[add]=[]

for add in os.listdir('annotations/'):
    ann_dict[add[0:4]+'.jpg'].append(add)

os.mkdir('global_seg')

for add in im_list:
    print add
    im0 = np.array(Image.open('images/'+add))
    im =  np.zeros(im0.shape[0:2])
    for ann_add in ann_dict[add]:
        ann = np.array(Image.open('annotations/'+ann_add))
        ann_b = ann > 0
        im = ann*ann_b + im * (1-ann_b)
    cv2.imwrite('global_seg/'+add.replace('.jpg', '.png'), im.astype(np.int))
    if 0:
        plt.figure(1)
        plt.imshow(im0)
        plt.figure(2)
        plt.imshow(im)
        plt.show()
