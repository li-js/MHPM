import torch
import torch.nn as nn
import numpy as np
import pickle
import cv2
from torch.autograd import Variable

in_margin = 0.5
out_margin= 1.5
Lnorm = 2


def myNorm(inp, L):
    if L==1:
        n = torch.sum(torch.abs(inp), 0, keepdim=True)
    else:
        n = torch.sqrt(torch.sum(torch.pow(inp, 2), 0, keepdim=True) + 1e-8)
    return n


def myInstanceLoss(preds, labels,ignore_index=None):
    '''
    preds:  batchsize x nDim x h x w
    labels: batchsize x 1 x h x w
    '''

    batchsize, C, _, _ = list(preds.size())

    if ignore_index is not None:
        preds_new, labels_new = [], []
        for k in xrange(batchsize):            
            mask = (labels[k][0] != ignore_index).float()
            w = int(mask[0].sum().data.cpu().numpy()[0])
            h = int(mask[:,0].sum().data.cpu().numpy()[0])
            labels_new.append(labels[k][0:1, 0:h, 0:w])
            preds_new.append(preds[k][:,0:h, 0:w])
        preds = preds_new
        labels= labels_new    

    loss = 0.0

    for b in xrange(batchsize):
        loss_dist = 0.0 
        loss_var  = 0.0

        pred  = preds[b]
        label = labels[b]

        means = []

        for j in xrange(int(torch.max(label).data.cpu().numpy()[0])):
            mask = label.eq(j+1)
            #mask_sum = torch.sum(mask)
            if torch.sum(mask.float()).data.cpu().numpy()[0] > 1:
                inst = pred[mask.expand_as(pred)].view(C,-1,1) #c x -1 x 1
                mean = torch.mean(inst, 1, keepdim=True)  # c x 1 x 1 
                means.append(mean)

                var = myNorm((inst - mean.expand_as(inst)), 2) # 1 x -1 x 1

                var = torch.clamp(var-in_margin, min=0)
                var = torch.pow(var,2).view(-1)
                var = torch.mean(var)
                loss_var = loss_var + var

        if len(means)>1:
            loss_d = 0.0
            for j in xrange(len(means)):
                mean_A = means[j]
                for k in xrange(j+1, len(means)):
                    mean_B = means[k]
                    d = myNorm(mean_A-mean_B, 2)  # 1 x 1 x 1
                    d = torch.pow(torch.clamp(2*out_margin-d, min=0), 2)
                    loss_d = loss_d + d.view(-1)[0]

            loss_dist = loss_dist + loss_d/(len(means)-1+1e-8)

        loss = loss + loss_var + loss_dist

    loss = loss/batchsize  #+ 0*torch.sum(preds) # use this term if the loss is 0.0. This is to stop the complain from autograd if there happens to be an image without any instances. In that case, no gradients would be calculated. By adding this extra term, the default gradient will be zero.

    return loss


def myInstanceLoss_group(preds, labels,ignore_index=None):
    '''
    preds:  batchsize x nDim x h x w
    labels: batchsize x 1 x h x w
    '''

    batchsize, C, _, _ = list(preds.size())

    if ignore_index is not None:
        preds_new, labels_new = [], []
        for k in xrange(batchsize):            
            mask = (labels[k][0] != ignore_index).float()
            w = int(mask[0].sum().data.cpu().numpy()[0])
            h = int(mask[:,0].sum().data.cpu().numpy()[0])
            labels_new.append(labels[k][0:1, 0:h, 0:w])
            preds_new.append(preds[k][:,0:h, 0:w])
        preds = preds_new
        labels= labels_new    

    loss = 0.0

    for b in xrange(batchsize):
        pred  = preds[b]
        label = labels[b]

        loss_push = 0.0
        loss_pull = 0.0

        means = []
        for j in xrange(int(torch.max(label).data.cpu().numpy()[0])):
            mask = label.eq(j+1)
            #mask_sum = torch.sum(mask)
            if torch.sum(mask.float()).data.cpu().numpy()[0] > 1:
                inst = pred[mask.expand_as(pred)].view(C,-1,1) #c x -1 x 1
                mean = torch.mean(inst, 1, keepdim=True)  # c x 1 x 1 
                means.append(mean)

                var = myNorm((inst - mean.expand_as(inst)), 2) # 1 x -1 x 1

                var = torch.clamp(var-in_margin, min=0)
                var = torch.pow(var,2).view(-1)
                var = torch.mean(var)
                loss_pull = loss_pull + var

        loss_pull = loss_pull/(len(means)+1e-8)

        if len(means)>1:
            for j in xrange(len(means)):
                mean_A = means[j]
                for k in xrange(j+1, len(means)):
                    mean_B = means[k]
                    d = myNorm(mean_A-mean_B, 2)  # 1 x 1 x 1
                    d = torch.pow(torch.clamp(2*out_margin-d, min=0), 2)
                    loss_push = loss_push + d.view(-1)[0]

        loss_push = loss_push*2/(len(means)*(len(means)-1)+1e-8)

        loss = loss + loss_pull + loss_push

    loss = loss/batchsize  #+ 0*torch.sum(preds) # use this term if the loss is 0.0. This is to stop the complain from autograd if there happens to be an image without any instances. In that case, no gradients would be calculated. By adding this extra term, the default gradient will be zero.

    return loss

def myInstanceLoss_individual(preds, labels,ignore_index=None, top_K=100):
    '''
    preds:  batchsize x nDim x h x w
    labels: batchsize x 1 x h x w
    '''
    
    batchsize, C, _, _ = list(preds.size())

    if ignore_index is not None:
        preds_new, labels_new = [], []
        for k in xrange(batchsize):            
            mask = (labels[k][0] != ignore_index).float()
            w = int(mask[0].sum().data.cpu().numpy()[0])
            h = int(mask[:,0].sum().data.cpu().numpy()[0])
            labels_new.append(labels[k][0:1, 0:h, 0:w])
            preds_new.append(preds[k][:,0:h, 0:w])
        preds = preds_new
        labels= labels_new    

    loss = 0.0

    for b in xrange(batchsize):
        pred  = preds[b]
        label = labels[b]

        tags_list = []
        for j in xrange(int(torch.max(label).data.cpu().numpy()[0])):
            mask = label.eq(j+1)
            if torch.sum(mask.float()).data.cpu().numpy()[0] > 1:
                inst = pred[mask.expand_as(pred)].view(C,-1) #c x -1 
                if top_K is not None:
                    tags = inst[:, np.random.permutation(inst.size(1))[0:top_K]]
                else:
                    tags = inst
                tags_list.append(tags)

        num_inst = len(tags_list)
        loss_pull = 0.0 
        loss_push = 0.0
        for tags in tags_list:
            tagsT = tags.transpose(1,0).contiguous()
            K_n = tagsT.size(0)
            tagsT_e1 = tagsT.view(K_n,1,C).repeat(1, K_n, 1)
            tagsT_e2 = tagsT.view(1,K_n,C).repeat(K_n, 1, 1)
            dists = torch.sqrt( torch.sum((tagsT_e1 - tagsT_e2)**2, dim=2) + 1e-8)
            dists_hinged = torch.clamp(dists - in_margin, min=0)
            loss_pull = loss_pull + torch.mean(dists_hinged**2)

        #loss_pull = loss_pull/num_inst         # Used before 2 Jun 2018
        loss_pull = loss_pull/(num_inst+1e-8)   # Changed on 2 Jun 2018
            
        if num_inst>1:
            for t_i in xrange(num_inst):
                for t_j in xrange(t_i+1, num_inst):
                    tags_i = tags_list[t_i].transpose(1,0).contiguous()
                    tags_j = tags_list[t_j].transpose(1,0).contiguous()
                    K_i, K_j = tags_i.size(0), tags_j.size(0)
                    tags_i_e = tags_i.view(K_i,1,C).repeat(1, K_j, 1)
                    tags_j_e = tags_j.view(1,K_j,C).repeat(K_i, 1, 1)
                    dists2 = torch.sqrt( torch.sum((tags_i_e - tags_j_e)**2, dim=2) + 1e-8)
                    dists2_hinged = torch.clamp(2*out_margin - dists2 , min=0)
                    loss_push = loss_push + torch.mean(dists2_hinged**2)
        #loss_push = loss_push*2/(num_inst*(num_inst-1))      # Used before 2 Jun 2018
        loss_push = loss_push*2/(num_inst*(num_inst-1)+1e-8)  # Changed on 2 Jun 2018

        loss = loss + loss_pull + loss_push

    loss = loss/batchsize  #+ 0*torch.sum(loss)

    return loss


if __name__ == '__main__':
    preds = Variable(torch.randn(2,8,3,4))
    labels= Variable(torch.round(torch.rand(2,1,3,4)*5))    
    loss = myInstanceLoss(preds, labels)
