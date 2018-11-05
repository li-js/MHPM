import sys, os
import numpy as np
import cv2
import pickle, gzip
from tqdm import tqdm, trange
#import scipy

DEBUG = 0
if DEBUG: from pylab import plt
from voc_eval import voc_ap
import evaluator

def get_gt(list_dat):
    class_recs = {}
    npos = 0

    for dat in tqdm(list_dat, desc='Loading gt..'):
        imagename = dat['filepath'].split('/')[-1]
        if len(dat['bboxes']) == 0:
            gt_box=np.array([])
            det = []
            anno_adds = []
        else:
            gt_box = []
            anno_adds = []
            for bbox in dat['bboxes']:
                mask_gt = cv2.imread(bbox['ann_path'], cv2.IMREAD_GRAYSCALE)
                if np.sum(mask_gt>0)==0: continue
                if np.allclose(mask_gt==255, mask_gt>0): continue # ignore label
                anno_adds.append(bbox['ann_path'])
                gt_box.append((bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']))
                npos = npos + 1 

            det = [False] * len(anno_adds)
        class_recs[imagename] = {'gt_box': np.array(gt_box),
                                 'anno_adds': anno_adds, 
                                 'det': det}
    return class_recs, npos

    
def eval_seg_ap(results_all, dat_list, nb_class=19, ovthresh_seg=0.4, SPARSE=False, From_pkl=False):
    confidence = []
    image_ids  = []
    BB = []
    Local_segs_ptr = []

    for imagename in tqdm(results_all.keys(), desc='Loading results ..'):
        if From_pkl:
            results = pickle.load(gzip.open(results_all[imagename]))
        else:
            results = results_all[imagename]

        det_rects = results['DETS']
        for idx, rect in enumerate(det_rects):
            image_ids.append(imagename)
            confidence.append(rect[-1])
            BB.append(rect[:4])
            Local_segs_ptr.append(idx)

    confidence = np.array(confidence)
    BB = np.array(BB)
    Local_segs_ptr = np.array(Local_segs_ptr)

    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    Local_segs_ptr = Local_segs_ptr[sorted_ind]
    image_ids =  [image_ids[x]  for x in sorted_ind]
        #return image_ids, BB

    #ovthresh_seg=0.4
    class_recs, npos = get_gt(dat_list)
    eva_local =evaluator.ParsingEval(nb_class=nb_class)
    nd = len(image_ids)
    tp_seg = np.zeros(nd)
    fp_seg = np.zeros(nd)
    pcp_list= []

    for d in trange(nd, desc='Finding AP^P at thres %f..'%ovthresh_seg):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        jmax = -1
        #BBGT = R['gt_box'].astype(float)
        #seg_local0 = results_all[image_ids[d]]['local_mask'][Local_segs_ptr[d]]
        if From_pkl:
            results = pickle.load(gzip.open(results_all[image_ids[d]]))
        else:
            results = results_all[image_ids[d]]

        mask0 = results['MASKS'][Local_segs_ptr[d]]
        if results['MASK_MODE'] == 'Fixed':
            bb_tmp = np.round(bb).astype(np.int)
            bb_tmp[bb_tmp<0] = 0
            bb_tmp[2] = bb_tmp[2]  if bb_tmp[2]<=results['WIDTH'] else results['WIDTH']
            bb_tmp[3] = bb_tmp[3]  if bb_tmp[3]<=results['HEIGHT'] else results['HEIGHT']
            mask_pred = np.zeros((results['HEIGHT'], results['WIDTH']), dtype=np.int)            
            mask_pred[bb_tmp[1]:bb_tmp[3], bb_tmp[0]:bb_tmp[2]]=cv2.resize(mask0, 
                                            (bb_tmp[2]-bb_tmp[0], bb_tmp[3]-bb_tmp[1])).argmax(axis=2)
        elif results['MASK_MODE'] == 'Original':
            if SPARSE:
                mask_pred = mask0.toarray().astype(np.int) # decode sparse array if it is one
            else:
                mask_pred = mask0.astype(np.int)
        else:
            assert(0)

        for i in xrange(len(R['anno_adds'])):
            mask_gt = cv2.imread(R['anno_adds'][i], cv2.IMREAD_GRAYSCALE)
            try:
                seg_iou= eva_local.cal_one_mean_iou(mask_pred.astype(np.uint8), mask_gt)
            except:
                seg_iou= eva_local.cal_one_mean_iou(mask_pred.astype(np.uint8), mask_gt)
            mean_seg_iou = np.nanmean(seg_iou)
            if mean_seg_iou > ovmax:
                ovmax =  mean_seg_iou
                seg_iou_max = seg_iou 
                jmax = i
                mask_gt_u = np.unique(mask_gt)
                #add_tmp = R['anno_adds'][i]


        if ovmax > ovthresh_seg:
            if not R['det'][jmax]:
                tp_seg[d] = 1.
                R['det'][jmax] = 1
                pcp_d = len(mask_gt_u[np.logical_and(mask_gt_u>0, mask_gt_u<nb_class)])
                pcp_n = float(np.sum(seg_iou_max[1:]>ovthresh_seg))
                if pcp_d > 0:
                    pcp_list.append(pcp_n/pcp_d)
                else:
                    pcp_list.append(0.0)
            else:
                fp_seg[d] =  1.
        else:
            fp_seg[d] =  1.

    # compute precision recall
    fp_seg = np.cumsum(fp_seg)
    tp_seg = np.cumsum(tp_seg)
    rec_seg = tp_seg / float(npos)
    prec_seg = tp_seg / (tp_seg + fp_seg)

    ap_seg = voc_ap(rec_seg, prec_seg)

    assert(np.max(tp_seg) == len(pcp_list)), "%d vs %d"%(np.max(tp_seg),len(pcp_list))
    #pcp_numerator = np.sum(pcp_list)
    pcp_list.extend([0.0]*(npos - len(pcp_list)))
    pcp = np.mean(pcp_list)

    print 'AP_seg, PCP:', ap_seg, pcp
    return ap_seg, pcp

def eval_seg_iou(results_all, dat_list, nb_class=19, SPARSE=False, From_pkl=False, cut_off_conf=0.0):
    # Evaluation of Global Segmentation and Evaluation of Global Segmentation based on local segments
    eva_global_iou = evaluator.ParsingEval(nb_class=nb_class)
    if 'mask_global' in results_all[dat_list[0]['filepath'].split('/')[-1]].keys():
        desc = 'Finding IOU by key mask_global..'
    else:
        desc = 'Finding IOU by combining ..'

    for dat in tqdm(dat_list, desc=desc):
        imagename = dat['filepath'].split('/')[-1]
        mask_gt = cv2.imread(dat['global_mask_add'], cv2.IMREAD_GRAYSCALE)

        if From_pkl:
            results = pickle.load(gzip.open(results_all[imagename]))
        else:
            results = results_all[imagename]

        if 'mask_global' in results.keys():
            if SPARSE:
                mask_global = results['mask_global'].toarray()
            else:
                mask_global = results['mask_global']
            mask_global = mask_global.astype(np.int)
        elif results['MASK_MODE'] == 'Fixed':
            mask_global =  np.zeros((mask_gt.shape[0], mask_gt.shape[1], nb_class), dtype=np.float32)  #make global mask from individual ones
            for idx, det in enumerate(results['DETS']):
                mask0 = results['MASKS'][idx]
                bb_tmp = np.round(det[0:4]).astype(np.int)
                bb_tmp[bb_tmp<0] = 0
                bb_tmp[2] = bb_tmp[2]  if bb_tmp[2]<=mask_gt.shape[1] else mask_gt.shape[1]
                bb_tmp[3] = bb_tmp[3]  if bb_tmp[3]<=mask_gt.shape[0] else mask_gt.shape[0]
                mask = cv2.resize(mask0, (bb_tmp[2]-bb_tmp[0], bb_tmp[3]-bb_tmp[1]))
                mask_global[bb_tmp[1]:bb_tmp[3], bb_tmp[0]:bb_tmp[2], :] += mask
            mask_global = mask_global.argmax(axis=2)
        elif results['MASK_MODE'] == 'Original':
            mask_global =  np.zeros((mask_gt.shape[0], mask_gt.shape[1]), dtype=np.int)  #make global mask from individual ones
            scores = [rect[-1] for rect in results['DETS']]
            indice = np.argsort(scores)
            for i in indice:
                if scores[i]<cut_off_conf: 
                    continue
                mask0 = results['MASKS'][i]
                if SPARSE:
                    mask0 = mask0.toarray() 
                mask_b = mask0>0
                mask_global = mask_global*(1- mask_b) + mask0*mask_b

        eva_global_iou.update(mask_global.astype(np.uint8), mask_gt)

    iou = eva_global_iou.get_mean_iou()
    print 'eva_global_iou:' , iou
    return iou


def get_prediction_from_gt(dat_list, NUM_CLASSES, use_parts=False, cache_pkl=False):
    results_all = {}
    for dat in tqdm(dat_list, desc='Generating predictions ..'):

        results = {} 
        results['WIDTH'], results['HEIGHT']=dat['width'], dat['height']

        dets = []
        masks= []
        if use_parts:
            dets_parts = {cls: [] for cls in xrange(1, NUM_CLASSES)}
            masks_parts= {cls: [] for cls in xrange(1, NUM_CLASSES)}    

        for box in dat['bboxes']:
            mask_gt = cv2.imread(box['ann_path'], cv2.IMREAD_GRAYSCALE)
            if np.sum(mask_gt)==0: continue
            #if np.allclose(mask_gt==255, mask_gt>0): continue # ignore label
            ys, xs = np.where(mask_gt>0)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            dets.append((x1, y1, x2, y2, 1.0))
            #mask_gt[mask_gt==255] = 0 # ignore label
            masks.append(mask_gt)
            if use_parts:
                for cls in np.unique(mask_gt):
                    if cls == 0: continue
                    if cls == 255: continue # ignore label
                    mask_b = mask_gt == cls
                    if np.sum(mask_b)==0: continue
                    ys, xs = np.where(mask_b)
                    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                    dets_parts[cls].append((x1, y1, x2, y2, 1.0))
                    masks_parts[cls].append(mask_b.astype(np.int))
            

        results['MASKS']= masks
        results['DETS'] = dets
        if use_parts:
            results['PART_DETS'] = dets_parts
            results['PART_MASKS']= masks_parts

        results['MASK_MODE'] ='Original'            
        key = dat['filepath'].split('/')[-1]
        if cache_pkl:
            results_cache_add = 'tmp/' + key + '.pklz'
            pickle.dump(results, gzip.open(results_cache_add, 'w'))
            results_all[key] = results_cache_add
        else:
            results_all[key]=results
    return results_all
