# encoding: utf-8
"""
@author: FroyoZzz
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: froyorock@gmail.com
@software: garner
@file: train.py
@time: 2019-08-05 13:46
@desc:
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import logging
from optparse import OptionParser
from model import fcn

CUDA = torch.cuda.is_available()

def train(**kwargs):
    model = kwargs["model"]
    loss = kwargs["loss"]
    data_loader = kwargs["data_loader"]
    optimizer = kwargs["optimizer"]
    epoch = kwargs["epoch"]
    save_freq = kwargs["save_freq"]
    save_dir = kwargs["save_dir"]
    verbose = kwargs["verbose"]

    logging.info("Epoch %03d, Learning Rate %g" % (epoch + 1, optimizer.param_groups[0]["lr"]))
    model.train()

    for i, (x, y) in enumerate(data_loader):
        pass



def main():
    parser = OptionParser()
    parser.add_option("-j", "--workers", dest="workers", default=4, type="int",
                      help="number of data loading workers (default: 4)")
    parser.add_option("-e", "--epochs", dest="epochs", default=80, type="int",
                      help="number of epochs (default: 80)")
    parser.add_option("-b", "--batch-size", dest="batch_size", default=16, type="int",
                      help="batch size (default: 16)")
    parser.add_option("-c", "--ckpt", dest="ckpt", default=False,
                      help="load checkpoint model (default: False)")
    parser.add_option("-v", "--verbose", dest="verbose", default=100, type="int",
                      help="show information for each <verbose> iterations (default: 100)")
    parser.add_option("-n", "--num-classes", dest="num_classes", default=20, type="int",
                      help="number of classes (default: 20)")
    parser.add_option("-d", "--back-bone", dest="back_bone", default="vgg",
                      help="backbone net (default: vgg)")

    parser.add_option("--lr", "--learn-rate", dest="lr", default=1e-2, type="float",
                      help="learning rate (default: 1e-2)")
    parser.add_option("--sf", "--save-freq", dest="save_freq", default=1, type="int",
                      help="saving frequency of .ckpt models (default: 1)")
    parser.add_option("--sd", "--save-dir", dest="save_dir", default="./models",
                      help="saving directory of .ckpt models (default: ./models)")
    parser.add_option("--init", "--initial-training", dest="initial_training", default=1, type="int",
                      help="train from 1-beginning or 0-resume training (default: 1)")

    (options, args) = parser.parse_args()

    mymodel = fcn.FCN32s(args.num_classes, args.back_bone)

    # checkpoint
    if args.ckpt:
        ckpt = args.ckpt

        if args.initial_training == 0:
            epoch_name = (ckpt.split('/')[-1]).split('.')[0]
            start_epoch = int(epoch_name)

        checkpoint = torch.load(ckpt)
        state_dict = checkpoint["state_dict"]

        mymodel.load_state_dict(state_dict)
        logging.info(f"Model loaded from {args.ckpt}")

    # initialize model-saving directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # CUDA
    if CUDA:
        mymodel.to(torch.device("cuda"))
        mymodel = nn.DataParallel(mymodel)

    # dataset


if __name__ == '__main__':

    main()