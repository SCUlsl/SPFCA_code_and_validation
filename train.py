# -*-coding:utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import UNetDataset
from nets.resunet import resunet
from nets.unet import unet
from nets.fcn import fcn
from nets.resunet import weights_init
from utils.train_utils import train_one_epoch
import pandas as pd
import shutil


def main(args):
    # set device to cuda if possible
    cuda = True if torch.cuda.is_available() else False
    # model
    model = resunet(
        in_channels=3,out_channels=1,depth=4,basewidth=32,drop_rate=0,
    )
    # model initialization
    weights_init(model)
    if cuda:
        torch.cuda.empty_cache
        model = model.cuda()

    # load model parameters 
    model_filename = os.path.join(args.training_log_dir,args.model_filename)
    if os.path.exists(model_filename):
        print('Load weights {}.'.format(model_filename))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_filename, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.6)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,  
    #                                                           min_lr=1e-8, eps=1e-08, verbose=False)



    # training datasets
    train_data_path = os.path.join(args.data_path, 'dataset1')
    train_dataset = UNetDataset(
        train=True,
        dataset_path=train_data_path,
        auto_label=True,
        auto_label_params={
            'n_segments':1000,
            'compactness':0.5,
        }
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    # validation datasets
    valid_data1_path = os.path.join(args.data_path, 'dataset1')
    valid_data2_path = os.path.join(args.data_path, 'dataset2')
    valid_data3_path = os.path.join(args.data_path, 'dataset3')
    valid_data4_path = os.path.join(args.data_path, 'dataset4')
    valid_dataset1 = UNetDataset(
        train=False,
        dataset_path=valid_data1_path
    )
    valid_dataset2 = UNetDataset(
        train=False,
        dataset_path=valid_data2_path
    )
    valid_dataset3 = UNetDataset(
        train=False,
        dataset_path=valid_data3_path
    )
    valid_dataset4 = UNetDataset(
        train=False,
        dataset_path=valid_data4_path
    )

    valid_loader1 = DataLoader(
        dataset=valid_dataset1,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader2 = DataLoader(
        dataset=valid_dataset2,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader3 = DataLoader(
        dataset=valid_dataset3,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader4 = DataLoader(
        dataset=valid_dataset4,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    training_log = list()
    # print("test training_log_filename:")
    # print(training_log_filename)
    logfile_name = os.path.join(args.training_log_dir, args.model_filename + f"lr{args.lr}train_loss.csv")
    if os.path.exists(logfile_name):
        training_log.extend(pd.read_csv(logfile_name).values)
        print(training_log)
        start_epoch = int(training_log[-1][0]) + 1
    else:
        start_epoch = 0
    training_log = list()
    training_log_header = ["epoch", "train_loss", "valid1_loss", "valid2_loss", "valid3_loss", "valid4_loss", "lr"]
    for epoch in range(start_epoch,args.sum_epoch):
        train_loss,valid1_loss,valid2_loss,valid3_loss,valid4_loss = train_one_epoch(model=model, train_loader=train_loader, valid_loader1=valid_loader1, valid_loader2=valid_loader2, valid_loader3=valid_loader3, valid_loader4=valid_loader4,
                                                optimizer=optimizer, epoch=epoch, lr=get_lr(optimizer),cuda=cuda)
        # update the training log
        training_log.append([epoch, train_loss, valid1_loss, valid2_loss, valid3_loss, valid4_loss, get_lr(optimizer)])
        pd.DataFrame(training_log, columns=training_log_header).set_index("epoch").to_csv(logfile_name)
        min_epoch = np.asarray(training_log)[:, training_log_header.index("valid1_loss")].argmin()
        # save model
        torch.save(model.state_dict(), model_filename)
        lr_scheduler.step()
        if  min_epoch == len(training_log) - 1:
            best_filename = model_filename.replace(".h5", "_best.h5")
            forced_copy(model_filename, best_filename)

        # if  (epoch % 10) == 0:
        #     epoch_filename = model_filename.replace(".h5", "_{}.h5".format(epoch))
        #     forced_copy(model_filename, epoch_filename)
    

def forced_copy(source, target):
    remove_file(target)
    shutil.copy(source, target)

def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)

def get_lr(optimizer):
    lrs = [params['lr'] for params in optimizer.param_groups]
    return np.squeeze(np.unique(lrs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--model_path', type=str, default='./logs')
    parser.add_argument('--data_path', type=str, default='./datasets')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sum_epoch', type=int, default=400)
    parser.add_argument('--training_log_dir', type=str, default='./logs')
    parser.add_argument('--model_filename', type=str)
    args = parser.parse_args()
    main(args)