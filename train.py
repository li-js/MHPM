import setGPU
import sys
sys.path.remove('/usr/lib/python2.7/dist-packages')
import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
#import matplotlib.pyplot as plt
import random
from tqdm import tqdm, trange
import pickle

sys.path.append('libs/')
from model_solver import Res_Deeplab
from model_solver import outS


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 1
IGNORE_LABEL = 255
INPUT_SIZE = '600,1000'
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_TAGS = 8
NUM_STEPS = 60000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'pre-trained/MHP_global_seg_pretrained.pth'
SAVE_PRED_EVERY = 3000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL).cuda(gpu)
    
    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

    b = []
    b.append(model.layer5.parameters())
    for j in range(len(b)):
        for i in b[j]:
            yield i

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer4_tag)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

    b = []
    b.append(model.layer5_tag.parameters())
    for j in range(len(b)):
        for i in b[j]:
            yield i
            
            
def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10


"""Create the model and start the training."""

h, w = map(int, args.input_size.split(','))
input_size = (h, w)

cudnn.enabled = True
gpu = args.gpu

model = Res_Deeplab(num_classes=NUM_CLASSES, num_tags=NUM_TAGS, split_layer=4)

saved_state_dict = torch.load(args.restore_from)
new_params = model.state_dict().copy()

for i in saved_state_dict:
    i_parts = i.split('.')

    if i_parts[1]=='layer4':
        print '.'.join(i_parts).replace('layer4', 'layer4_tag')
        assert(new_params['.'.join(i_parts).replace('layer4', 'layer4_tag')].size() == saved_state_dict[i].size())
        new_params['.'.join(i_parts).replace('layer4', 'layer4_tag')] = saved_state_dict[i]

    print '.'.join(i_parts)
    assert(new_params['.'.join(i_parts)].size() == saved_state_dict[i].size())
    new_params['.'.join(i_parts)] = saved_state_dict[i]

c1=c2=0
for i in new_params:
    i_parts = i.split('.')
    if i_parts[0]=='layer4':
        c1+=1
    if i_parts[0]=='layer4_tag':
        c2+=1

assert(c1==c2)
    
model.load_state_dict(new_params)

#model.float()
model.eval() # use_global_stats = True
#model.train()
model.cuda(args.gpu)

cudnn.benchmark = False

if not os.path.exists(args.snapshot_dir):
    os.makedirs(args.snapshot_dir)

cache_add = 'cache_dat_list.pkl'
if os.path.isfile(cache_add):
    lists = pickle.load(open(cache_add))
    train_dat_list = lists['train']
    test_dat_list  = lists['test']
    val_dat_list   = lists['val']
    for dat in train_dat_list: dat['global_tag_add'] = dat['global_mask_add'].replace('global_seg/','global_tag/')
else:
    import mhp_data
    train_dat_list= mhp_data.get_train_dat()
    val_dat_list  = mhp_data.get_val_dat()
    test_dat_list = mhp_data.get_test_dat()
    lists = {'test': test_dat_list, 'val': val_dat_list, 'train': train_dat_list}
    for dat in train_dat_list: dat['global_tag_add'] = dat['global_mask_add'].replace('global_seg/','global_tag/')
    pickle.dump(lists, open(cache_add,'w'))

    
import datasets
import myInstanceLoss

#pre_scale = None
pre_scale = (600,1000)
#crop_size = (321, 321)
crop_size = None


trainloader = torch.utils.data.DataLoader(
            datasets.MHPGlobalDataSet(train_dat_list, mean=IMG_MEAN, pre_scale=pre_scale, crop_size=crop_size, 
                with_tag=True, tag_scaling=outS), 
            batch_size=BATCH_SIZE, shuffle=True, num_workers=5, pin_memory=True)

optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.learning_rate }, 
            {'params': get_10x_lr_params(model), 'lr': 10*args.learning_rate}], 
            lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)

for idx,pg in enumerate(optimizer.param_groups):
    print 'Got %d parameters in params group %d with lr %f'%(len(pg['params']), idx, pg['lr'])

optimizer.zero_grad()

#interp = nn.Upsample(size=input_size, mode='bilinear')

EPOCH_num = args.num_steps*BATCH_SIZE/len(train_dat_list)
BATCH_num = len(train_dat_list)/BATCH_SIZE
pbar =  trange(args.num_steps)

for epoch_num in xrange(EPOCH_num):
    pbar.set_description('Training Epoch {}/{} ({} batches) ..'.format(epoch_num, EPOCH_num, BATCH_num))
    for i_iter, batch in enumerate(trainloader):
        images = Variable(batch['image']).cuda(args.gpu)

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter+epoch_num*BATCH_num)
        interp = nn.Upsample(size=batch['label'].size()[1:3], mode='bilinear')
        seg, tag = model(images)
        seg_big = interp(seg)
        #tag_big = interp(tag)
        loss_seg = loss_calc(seg_big, batch['label'], args.gpu)
        tag_target = Variable(batch['label_tag'].unsqueeze(1)).cuda()
        loss_tag_grp = myInstanceLoss.myInstanceLoss_group(tag, tag_target, ignore_index=None)
        loss_tag_ind = myInstanceLoss.myInstanceLoss_individual(tag, tag_target, ignore_index=None)
        loss_tag = loss_tag_grp + loss_tag_ind
        loss = loss_seg + loss_tag
        loss.backward()
        optimizer.step()

        pbar.update()
        pbar.set_postfix(loss_seg =loss_seg.data.cpu().numpy(), loss_tag_grp =loss_tag_grp.data.cpu().numpy(),
                         loss_tag_ind =loss_tag_ind.data.cpu().numpy(), lr=optimizer.param_groups[0]['lr'])
    #print 'iter = ', i_iter, 'of', args.num_steps,'completed, loss = ', loss.data.cpu().numpy()

    print 'taking snapshot ...'
    torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'MHP_'+str(epoch_num)+'.pth'))     


