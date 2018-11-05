import sys
import argparse
import torch   # torch.__version__: '0.3.1'
import torch.nn as nn 
import numpy as np # np.__version__: '1.8.2'
import pickle, gzip
import cv2         # cv2.__version__: '2.4.8'
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
import random
from tqdm import tqdm, trange
import pickle

sys.path.append('libs/')
from model_solver import Res_Deeplab
import datasets
import clustering


import pydensecrf.densecrf as dcrf   # From https://github.com/lucasb-eyer/pydensecrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 1
IGNORE_LABEL = 255
INPUT_SIZE = '600,1000'
NUM_CLASSES = 19
NUM_TAGS = 8

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='val', choices=['val', 'test']) 
parser.add_argument("--hard_set", type=str, default=None, choices=[None, 'top20', 'top5']) 
parser.add_argument("--trained_model", type=str, default='models/models-MHPM-SD2_plus.pth')
parser.add_argument("--refine_global", type=int, default=1)   # Use CRF to refine the global segmentation or not, default is True
parser.add_argument("--plot", type=int, default=0)
parser.add_argument("--cluster_delta", type=int, default=1.5)
args = parser.parse_args()
PLOT = args.plot

out_dir = 'cache_eval_refined_%s_%s_delta%f/'%((args.dataset, args.trained_model.replace('/','_'), args.cluster_delta)) # Cache the results

if args.refine_global:
    out_dir=out_dir.replace('_refined_', '_refined_global_')

if not os.path.isdir(out_dir): os.makedirs(out_dir)
out_add = out_dir + 'results.pkl'

if args.hard_set is not None:
    out_add = out_add.replace('.pkl', '_%s.pkl'%args.hard_set)
    hard_names_add = 'data/MHP_v1_hard_lists/hard_list_%s_%s.txt'%(args.dataset, args.hard_set)
    hard_names = [f.strip() for f in open(hard_names_add).readlines()]    

print out_add

h, w = map(int, INPUT_SIZE.split(','))
input_size = (h, w)

cudnn.enabled = True
gpu = 0

# Create network.
model = Res_Deeplab(num_classes=NUM_CLASSES, num_tags=NUM_TAGS, split_layer=4)

saved_state_dict = torch.load(args.trained_model)
model.load_state_dict(saved_state_dict)

model.eval() 
model.cuda(0)
cudnn.benchmark = False

cache_add = 'cache_dat_list.pkl'
if os.path.isfile(cache_add):
    lists = pickle.load(open(cache_add))
    train_dat_list = lists['train']
    test_dat_list  = lists['test']
    val_dat_list   = lists['val']
else:
    import mhp_data
    train_dat_list= mhp_data.get_train_dat()
    val_dat_list  = mhp_data.get_val_dat()
    test_dat_list = mhp_data.get_test_dat()
    lists = {'test': test_dat_list, 'val': val_dat_list, 'train': train_dat_list}
    pickle.dump(lists, open(cache_add,'w'))

dat_list = lists[args.dataset]

if args.hard_set is not None:
    dat_list_hard = []
    for dat in tqdm(dat_list, desc='Geting hard subset ..'):
        if dat['filepath'].split('/')[-1] in hard_names:
            dat_list_hard.append(dat)

    dat_list = dat_list_hard


testloader = torch.utils.data.DataLoader(datasets.MHPGlobalTestDataSet(dat_list, mean=IMG_MEAN, target_size=input_size), 
                batch_size=BATCH_SIZE, shuffle=False, num_workers=5, pin_memory=True)


def get_det_mask(seg_prob, instance_map, compat=0):
    dets_ = []
    masks_b_ = []
    for pidx in np.unique(instance_map):
        if pidx == 0: continue
        seg_map_b = (instance_map==pidx)
        ys, xs = np.where(seg_map_b)
        try:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            score = 1-(seg_prob[:,:,0]*seg_map_b).sum()/float(seg_map_b.sum())
        except:
            x1, y1, x2, y2, score_sp=0,0,0,0,0
        dets_.append( np.array((x1,y1,x2,y2, score)))
        masks_b_.append(seg_map_b)
    return dets_, masks_b_

min_width, min_height, min_area, min_box = 43, 85, 4600, 7480
def filter_small(dets, masks):
    dets2 = []
    masks2 = []
    for d, m in zip(dets, masks):
        mb = m>0
        if np.sum(mb)<min_area: continue
        ys, xs = np.where(mb)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        width = x2-x1
        height= y2-y1
        if width<min_width or height<min_height or (width*height)<min_box: continue
        dets2.append(d)
        masks2.append(m)
    #if len(dets2)<len(dets): print('Removing persons: {}->{}'.format(len(dets), len(dets2)))
    return dets2, masks2


sm = torch.nn.Softmax2d()
pbar = trange(len(dat_list), desc='Eval MHP with Tag..')

i_iter_too_big_on_gpu = []  #[97, 226, 371, 646, 784, 797, 835, 850]
results_all ={}
for i_iter, batch in enumerate(testloader):
    key = batch['filepath'][0].split('/')[-1]
    if os.path.isfile(out_dir+key+'.pkz'):                
        results_all[key] = pickle.load(gzip.open(out_dir+key+'.pkz'))
        pbar.update()
        continue

    if i_iter in i_iter_too_big_on_gpu:
        images = Variable(batch['image'])
        model.cpu()
        model.eval()
        out_seg, out_tag = model(images)
        model.cuda()
        model.eval()
    else:
        images = Variable(batch['image']).cuda(0)
        out_seg, out_tag = model(images)

    h_original, w_original = batch['label'].shape[1:3]    

    interp = torch.nn.Upsample(size=(h_original, w_original), mode='bilinear')
    seg_prob = interp(sm(out_seg)).data[0].cpu()
    out_tag  = interp(out_tag).data[0].cpu()

    if args.refine_global:
        feats_spatial = create_pairwise_gaussian(sdims=(3, 3), shape=(h_original, w_original))
        feats_bilateral = create_pairwise_bilateral(sdims=(20, 20), schan=(3, 3, 3), img=cv2.resize((batch['image'][0].numpy().transpose(1,2,0)+IMG_MEAN), (w_original, h_original)).astype(np.uint8), chdim=2)
        d = dcrf.DenseCRF2D(w_original, h_original, NUM_CLASSES) 
        U = -np.log(seg_prob.numpy()).reshape((NUM_CLASSES,-1)).astype(np.float32).copy(order='C')
        d.setUnaryEnergy(U)

        d.addPairwiseEnergy(feats_spatial, compat=3)
        d.addPairwiseEnergy(feats_bilateral, compat=4)

        Q = d.inference(10)
        #seg_map_crf = np.argmax(Q, axis=0).reshape((im_height,im_width))
        seg_prob = np.array(Q).reshape((NUM_CLASSES,h_original,w_original))
        seg_prob = torch.from_numpy(seg_prob)

    _, seg_cls = torch.max(seg_prob, 0, keepdim=True)
    instance_map_torch = clustering.clustering(out_tag, seg_cls, 1.5, perform_opening=True)


    seg_prob = seg_prob.numpy().transpose((1,2,0))
    results = {}    
    results['mask_global'] = seg_prob.argmax(axis=2)
    results['dets'], results['masks_b'] = get_det_mask(seg_prob, instance_map_torch.numpy())

    results['DETS'], results['MASKS'] =  filter_small(
                results['dets'], [m*results['mask_global'] for m in results['masks_b']])
    results['MASK_MODE'] = 'Original'

    key = batch['filepath'][0].split('/')[-1]
    pickle.dump(results, gzip.open(out_dir+key+'.pkz', 'w'))

    results_all[key] = results

    pbar.update()

    if PLOT:
        from pylab import plt
        plt.figure('im')
        plt.imshow( (batch['image'].numpy()[0].transpose((1,2,0))+IMG_MEAN).astype(np.uint8)[:,:,::-1])
        plt.figure('Seg')
        plt.imshow(results['mask_global'])
        plt.figure('With erode/dilate torch')
        plt.imshow(instance_map_torch.numpy())
        plt.show()

sys.path.append('libs/evaluate')
import eval_metrics_v2

#iou = eval_metrics_v2.eval_seg_iou(results_all, dat_list, nb_class=19)
#final_results = {'iou': iou}

ap_list, pcp_list = [], []
for thr in [float(k)/10 for k in xrange(1, 10)]:
    ap_, pcp_ = eval_metrics_v2.eval_seg_ap(results_all, dat_list, ovthresh_seg=thr, From_pkl=False)
    print 'Done for threshold: {}, ap: {}, pcp: {}'.format(thr, ap_, pcp_)
    ap_list.append(ap_)
    pcp_list.append(pcp_)

final_results.update({'ap_list': ap_list,  'pcp_list': pcp_list})

pickle.dump(final_results, open(out_add, 'w'))

data = pickle.load(open(out_add))
print '='*30+out_add+'='*30
print 'Ap \t\t Ap(val) \t PCP'
print '{:0.2f}\t\t{:0.2f}\t\t{:0.2f}\t\t{}'.format(
    100*data['ap_list'][4], 100*np.mean(data['ap_list']), 100*data['pcp_list'][4], out_add)