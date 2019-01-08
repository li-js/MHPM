import os
import cv2
from PIL import Image
import numpy as np
#from pylab import plt
ann_dict={}
im_list=sorted(os.listdir('images/'))
for add in im_list:
    ann_dict[add]=[]

for add in os.listdir('annotations/'):
    ann_dict[add[0:4]+'.jpg'].append(add)

os.mkdir('global_tag')
for add in im_list:
    print add
    #im0 = np.array(Image.open('images/'+add))
    im0 = cv2.imread('images/'+add)[:,:,::-1]
    tag = np.zeros(im0.shape[0:2])

    for idx, ann_add in enumerate(ann_dict[add]):
        ann = np.array(Image.open('annotations/'+ann_add))
        tag[ann>0] = idx+1

    cv2.imwrite('global_tag/'+add.replace('.jpg', '.png'), tag.astype(np.int))
    if 0:
        plt.figure(1)
        plt.imshow(im0)
        plt.figure(2)
        plt.imshow(tag)
        plt.figure(3)
        plt.imshow(cv2.imread('global_tag/'+add.replace('.jpg', '.png'), cv2.IMREAD_GRAYSCALE))
        plt.show()

