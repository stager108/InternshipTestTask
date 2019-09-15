
import torch
import torch.nn as nn
import torch.utils.data as dt
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
from tqdm import *
import numpy as np
import matplotlib.pyplot as plt

from model import SegmenterModel
from image_dataset import *
from metrics import *

useCuda = True
n_epoch = 15
log = '/content/drive/My Drive/laba/log'
train = '/content/drive/My Drive/laba/data/train/'
train_masks = '/content/drive/My Drive/laba/data/train_mask'
test = '/content/drive/My Drive/laba/data/valid/'
test_masks = '/content/drive/My Drive/laba/data/valid_mask'

batch_size = 5
segm_model = SegmenterModel()


def dice_loss(pred, target, smooth = 0.01):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

if __name__ == '__main__':
    criterion = nn.MSELoss()
    optimizer = optim.Adam(segm_model.parameters(), lr=0.0001)
    lr = 0.0001
    if useCuda == True:
        segm_model = segm_model.cuda()
        criterion= criterion.cuda()

    ds = ImageDataset(train, train_masks, augment=True)
    ds_test = ImageDataset(test, test_masks, augment = False)

    dl      = dt.DataLoader(ds, shuffle=True, num_workers=4, batch_size=batch_size)
    dl_test = dt.DataLoader(ds_test, shuffle=False, num_workers=4, batch_size=1)

    global_iter = 0
    for epoch in range(0, n_epoch):
        print ("Current epoch: ", epoch)
        epoch_loss = 0
        epoch_dice = 0
        segm_model.train(True)
        for iter, (cur_input, cur_target) in enumerate(tqdm( dl) ):
            cur_input = Variable(cur_input)
            cur_target = Variable(cur_target)
            if useCuda :
                cur_input = cur_input.cuda()
                cur_target = cur_target.cuda()
            cur_output = segm_model(cur_input)
            loss =  dice_loss(cur_output.type(torch.float), cur_target)
            loss.backward()
            optimizer.step()

            global_iter += 1
            epoch_loss += loss.item()
            cur_output = (cur_output >= 0.5).type(torch.float)
            epoch_dice += get_dice(cur_output.cpu().detach().numpy(), cur_target.cpu().detach().numpy())
        epoch_loss = epoch_loss / float(len(ds))
        epoch_dice = epoch_dice / float(len(ds))*batch_size
        print ("Epoch loss", epoch_loss)
        print ("Epoch dice", epoch_dice)
      #  tb_writer.add_scalar('Loss/Train', epoch_loss, epoch)

        print ("Make test")
        test_loss = 0
        test_dice = 0
        segm_model.train(False)

        for iter, (cur_input, cur_target) in enumerate(tqdm(dl_test)):
            cur_input = Variable(cur_input, volatile = True)
            cur_target = Variable(cur_target, volatile = True)
            if useCuda :
                cur_input = cur_input.cuda()
                cur_target = cur_target.cuda()
            cur_output = segm_model(cur_input)
            #print(o)
            loss = dice_loss(cur_output.type(torch.float), cur_target)
            test_loss += loss.item()
            cur_output = (cur_output >= 0.5).type(torch.float)
            test_dice += get_dice(cur_output.cpu().detach().numpy(), cur_target.cpu().detach().numpy())

        test_loss = test_loss / float(len(ds_test))
        test_dice = test_dice / float(len(ds_test))
        print ("Test loss", test_loss)
        
        print ("Test dice", test_dice)
        
        if epoch == 30:
            lr = lr/10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
    torch.save(segm_model.state_dict(), '/content/drive/My Drive/laba/segm_model1.pth')
