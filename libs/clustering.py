import numpy as np
import torch
from torch.autograd import Variable
import cv2

Lnorm = 2

def myNorm(inp, L):
    if L==1:
        n = torch.sum(torch.abs(inp), 0, keepdim=True)
    else:
        n = torch.sqrt(torch.sum(torch.pow(inp, 2), 0, keepdim=True) + 1e-8)
    return n


def meanshift(samples, mean, bandwidth):
    ndim = samples.size()[0]
    norm_map =  myNorm(samples - mean.expand_as(samples), 2)
    mask =  torch.lt(norm_map, bandwidth).expand_as(samples)
    if mask.sum()>0:
        new_mean = torch.mean(samples[mask].view(ndim, -1), 1, keepdim=True)
    else:
        new_mean =  mean
    return new_mean


def erode_and_dilate_cv2(mask_):
    mask_ = mask_.numpy()
    kernel = np.ones((3,3), np.uint8)
    mask_ = cv2.erode(mask_, kernel, iterations=1)
    mask_ = cv2.dilate(mask_, kernel, iterations=1)
    return torch.Tensor(mask_.astype(np.int)).byte()

def erode_and_dilate_torch(mask_):
    
    kern = torch.ones(3,3).type_as(mask_)
    hpad = int(kern.size(0)/2.0-0.5)
    wpad = int(kern.size(1)/2.0-0.5)
    pad = 1

    h = mask_.size(0)+2*hpad
    w = mask_.size(1)+2*wpad
    padded = (pad*torch.ones(h, w)).type_as(mask_)
    padded [hpad:mask_.size(0)+hpad, wpad:mask_.size(1)+wpad] = mask_

    n = kern.sum()
    mask_eroded = torch.nn.functional.conv2d(
            Variable(padded.view(1,1,h,w).float()), Variable(kern.view(1,1,3,3).float()))[0][0].data.eq(n).type_as(mask_)

    pad = 0
    padded = (pad*torch.ones(mask_.size(0)+2*hpad, mask_.size(1)+2*wpad)).type_as(mask_)
    padded [hpad:mask_.size(0)+hpad, wpad:mask_.size(1)+wpad] = mask_eroded

    mask_eroded_dilated = torch.nn.functional.conv2d(
         Variable(padded.view(1,1,h,w).float()), Variable(kern.view(1,1,3,3).float()))[0][0].data.gt(0).type_as(mask_)

    return mask_eroded_dilated


def clustering(tags, segs, bandwidth, perform_opening=True, use_torch=True):
    '''
    Expect Torch tensors:
    tags: 1 x nDim x h x w
    segs: 1 x 1 x h x w
    '''    
    if len(tags.size()) == 4: tags = tags[0]
    if len(segs.size()) == 4: segs = segs[0]

    ndim, h, w = list(tags.size())

    mask = segs.ne(0).view(-1)
    if mask.float().sum() == 0:
        return torch.zeros([1,h,w]).byte()

    tags = tags.view(ndim, -1)

    unclustered  = torch.ones(h*w).byte()*mask
    instance_map = torch.zeros(h*w).int()

    l = 1
    counter = 0
    ITER_MAX = 100
    ITER_C = 0

    while unclustered.sum()>100 and counter < 20:
        ITER_C = ITER_C + 1
        if ITER_C>ITER_MAX: 
            #print 'Max iteration reached during clusting!!'
            break

        tags_masked = tags[unclustered.view(1, -1).expand_as(tags).bool()].view(ndim, -1)

        index = np.random.randint(0, tags_masked.size()[1])
        mean =  tags_masked[:, index:index+1]

        new_mean = meanshift(tags_masked, mean, bandwidth)
        it = 0
        while torch.norm(mean-new_mean) > 0.0001 and it < 100:
            mean = new_mean
            new_mean =  meanshift(tags_masked, mean, bandwidth)
            it = it + 1

        if it < 100:
            norm_map = myNorm(tags - new_mean.expand_as(tags),2)

            th_mask = torch.lt(norm_map, bandwidth).view(-1)

            # iou = (instance_map.gt(0)*th_mask).sum()/(th_mask).sum()
            iou = (instance_map.gt(0)*th_mask).sum()/(th_mask).sum().numpy()

            if iou < 0.5:
                th_mask = th_mask*unclustered

                if perform_opening: 
                    if use_torch:
                        th_mask = erode_and_dilate_torch(th_mask.view(h,w))
                    else:
                        th_mask = erode_and_dilate_cv2(th_mask.view(h,w))

                instance_map[th_mask.view(-1).bool()] = l
                l = l + 1

            unclustered[th_mask.view(-1).bool()] = 0
            counter = 0
        else:
            counter = counter + 1

    tmp = torch.zeros(h*w).byte()
    l = 1
    for j in range(1, torch.max(instance_map)+1):
        if (instance_map.eq(j).sum()>50):
            tmp[instance_map.eq(j)] = l
            l = l + 1

    instance_map = tmp.view(h , w)

    return instance_map


if __name__ == '__main__':
    tags = torch.randn(1,8,3,4)
    segs = torch.round(torch.rand(1,1,3,4)*5).eq(2)
    instance_map = clustering(tags, segs, 1.5)    


    idx = np.random.randint(3000, 4980)
    segs = np.load('cache_preds/%04d.jpg_seg_prob.npy'%idx).argmax(axis=0)>0
    tags = np.load('cache_preds/%04d.jpg_tag.npy'%idx)

    segs = torch.Tensor(segs.astype(np.int)).byte()
    tags = torch.Tensor(tags).float()
    
    instance_map_torch = clustering(tags, segs, 1.5, perform_opening=True)

    instance_map_cv2 = clustering(tags, segs, 1.5, perform_opening=True, use_torch=False)

    instance_map_no = clustering(tags, segs, 1.5, perform_opening=False)

    instance_map_gt = np.load('cache_preds/%04d.jpg_ins_small.npy'%idx)
    instance_map_gt2 = np.load('cache_preds/%04d.jpg_ins.npy'%idx)

    from pylab import plt

    plt.figure('Seg')
    plt.imshow(segs.numpy())

    plt.figure('Without erode/dilate')
    plt.imshow(instance_map_no.numpy())


    plt.figure('With erode/dilate torch')
    plt.imshow(instance_map_torch.numpy())

    plt.figure('With erode/dilate cv2')
    plt.imshow(instance_map_cv2.numpy())

    plt.figure('GT')
    plt.imshow(instance_map_gt[0])    

    plt.figure('GT big')
    plt.imshow(instance_map_gt2[0])        

    plt.figure('GT*mask')
    plt.imshow(instance_map_gt[0] *(segs.numpy()>0))   

    plt.show()