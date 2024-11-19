"""
Modified from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils.loss_utils import Dice_Loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, train_loader,valid_loader1, valid_loader2, valid_loader3, valid_loader4, optimizer, epoch, lr, cuda):
    unet_time = AverageMeter('UNetTime', ':6.3f')
    train_losses = AverageMeter('train_Loss', ':.4e')
    valid1_losses = AverageMeter('valid1_Loss', ':.4e')
    valid2_losses = AverageMeter('valid2_Loss', ':.4e')
    valid3_losses = AverageMeter('valid3_Loss', ':.4e')
    valid4_losses = AverageMeter('valid4_Loss', ':.4e')
    lr_ = AverageMeter('lr', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [unet_time, train_losses, valid1_losses, valid2_losses, valid3_losses, valid4_losses, lr_],
        prefix="epoch: [{}]".format(epoch))
    lr_.update(lr)
    # switch to train mode
    model.train()
    end = time.time()
    for iteration,train_batch in enumerate(train_loader):
        train_data,train_target = train_batch
        if cuda:
            torch.cuda.empty_cache()
        optimizer.zero_grad()
        train_loss = train(model=model,data=train_data,target=train_target,cuda=cuda)
        # compute gradient and do step
        train_loss.backward()
        optimizer.step() 
        # measure accuracy and record loss
        train_losses.update(train_loss.item())         
    model.eval()
    for valid_batch in valid_loader1:
        valid_data,valid_target = valid_batch
        if cuda:
            torch.cuda.empty_cache()
        valid1_loss = valid(model=model,data=valid_data,target=valid_target,cuda=cuda)
        valid1_losses.update(valid1_loss.item())
    for valid_batch in valid_loader2:
        valid_data,valid_target = valid_batch
        if cuda:
            torch.cuda.empty_cache()
        valid2_loss = valid(model=model,data=valid_data,target=valid_target,cuda=cuda)
        valid2_losses.update(valid2_loss.item())
    for valid_batch in valid_loader3:
        valid_data,valid_target = valid_batch
        if cuda:
            torch.cuda.empty_cache()
        valid3_loss = valid(model=model,data=valid_data,target=valid_target,cuda=cuda)
        valid3_losses.update(valid3_loss.item())
    for valid_batch in valid_loader4:
        valid_data,valid_target = valid_batch
        if cuda:
            torch.cuda.empty_cache()
        valid4_loss = valid(model=model,data=valid_data,target=valid_target,cuda=cuda)
        valid4_losses.update(valid4_loss.item())
    unet_time.update(time.time() - end)
    progress.display(iteration)
    return train_losses.avg,valid1_losses.avg,valid2_losses.avg,valid3_losses.avg,valid4_losses.avg

                          

def train(model, data, target, cuda):
    # training
    model.train()
    imgs, labs = data, target
    if cuda:
        torch.cuda.empty_cache()
        imgs = imgs.cuda()
        labs = labs.cuda()
    outputs = model(imgs)
    criterion = Dice_Loss
    loss = criterion(outputs, labs)
    return loss

def valid(model, data, target, cuda):
    # training
    model.eval()
    imgs, labs = data, target
    with torch.no_grad():
        if cuda:
            torch.cuda.empty_cache()
            imgs = imgs.cuda()
            labs = labs.cuda()
        outputs = model(imgs)
        criterion = Dice_Loss
        loss = criterion(outputs, labs)
    return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

