#!/usr/bin/env python

import argparse
import os
import sys
import torch
import torch.nn as nn

import datasets
import models.resnet as ResNet
import models.senet as SENet
from trainer import Trainer, Validator
from extractor import Extractor
import utils

configurations = {
    1: dict(
        max_iteration=1000000,
        lr=1.0e-1,
        momentum=0.9,
        weight_decay=0.0,
        gamma=0.1, # "lr_policy: step"
        step_size=1000000, # "lr_policy: step"
        interval_validate=1000,
    ),
}

def get_parameters(model, bias=False):
    for k, m in model._modules.items():
        if k == "fc" and isinstance(m, nn.Linear):
            if bias:
                yield m.bias
            else:
                yield m.weight

N_IDENTITY = 8631  # the number of identities in VGGFace2 for which ResNet and SENet are trained

def main():
    parser = argparse.ArgumentParser("PyTorch Face Recognizer")
    parser.add_argument('cmd', type=str,  choices=['train', 'test', 'extract'], help='train, test or extract')
    parser.add_argument('--arch_type', type=str, default='resnet50_ft', help='model type',
                        choices=['resnet50_ft', 'senet50_ft', 'resnet50_scratch', 'senet50_scratch'])
    parser.add_argument('--dataset_dir', type=str, default='/path/to/dataset_directory', help='dataset directory')
    parser.add_argument('--log_file', type=str, default='/path/to/log_file', help='log file')
    parser.add_argument('--train_img_list_file', type=str, default='/path/to/train_image_list.txt',
                        help='text file containing image files used for training')
    parser.add_argument('--test_img_list_file', type=str, default='/path/to/test_image_list.txt',
                        help='text file containing image files used for validation, test or feature extraction')
    parser.add_argument('--meta_file', type=str, default='/path/to/identity_meta.csv', help='meta file')
    parser.add_argument('--checkpoint_dir', type=str, default='/path/to/checkpoint_directory',
                        help='checkpoints directory')
    parser.add_argument('--feature_dir', type=str, default='/path/to/feature_directory',
                        help='directory where extracted features are saved')
    parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys(),
                        help='the number of settings and hyperparameters used in training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--resume', type=str, default='', help='checkpoint file')
    parser.add_argument('--weight_file', type=str, default='/path/to/weight_file.pkl', help='weight file')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--horizontal_flip', action='store_true', 
                        help='horizontally flip images specified in test_img_list_file')
    args = parser.parse_args()
    print(args)

    if args.cmd == "extract":
        utils.create_dir(args.feature_dir)

    if args.cmd == 'train':
        utils.create_dir(args.checkpoint_dir)
        cfg = configurations[args.config]

    log_file = args.log_file
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    if cuda:
        print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 0. id label map
    meta_file = args.meta_file
    id_label_dict = utils.get_id_label_map(meta_file)

    # 1. data loader
    root = args.dataset_dir
    train_img_list_file = args.train_img_list_file
    test_img_list_file = args.test_img_list_file

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}

    if args.cmd == 'train':
        dt = datasets.VGG_Faces2(root, train_img_list_file, id_label_dict, split='train')
        train_loader = torch.utils.data.DataLoader(dt, batch_size=args.batch_size, shuffle=True, **kwargs)

    dv = datasets.VGG_Faces2(root, test_img_list_file, id_label_dict, split='valid',
                             horizontal_flip=args.horizontal_flip)
    val_loader = torch.utils.data.DataLoader(dv, batch_size=args.batch_size, shuffle=False, **kwargs)

    # 2. model
    include_top = True if args.cmd != 'extract' else False
    if 'resnet' in args.arch_type:
        model = ResNet.resnet50(num_classes=N_IDENTITY, include_top=include_top)
    else:
        model = SENet.senet50(num_classes=N_IDENTITY, include_top=include_top)
    # print(model)

    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
        assert checkpoint['arch'] == args.arch_type
        print("Resume from epoch: {}, iteration: {}".format(start_epoch, start_iteration))
    else:
        utils.load_state_dict(model, args.weight_file)
        if args.cmd == 'train':
            model.fc.reset_parameters()

    if cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()

    # 3. optimizer
    if args.cmd == 'train':
        optim = torch.optim.SGD(
            [
                {'params': get_parameters(model, bias=False)},
                {'params': get_parameters(model, bias=True), 'lr': cfg['lr'] * 2, 'weight_decay': 0},
            ],
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])
        if resume:
            optim.load_state_dict(checkpoint['optim_state_dict'])
    
        # lr_policy: step
        last_epoch = start_iteration if resume else -1
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim,  cfg['step_size'],
                                                       gamma=cfg['gamma'], last_epoch=last_epoch)

    if args.cmd == 'train':
        trainer = Trainer(
            cmd=args.cmd,
            cuda=cuda,
            model=model,
            criterion=criterion,
            optimizer=optim,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            log_file=log_file,
            max_iter=cfg['max_iteration'],
            checkpoint_dir=args.checkpoint_dir,
            print_freq=1,
        )
        trainer.epoch = start_epoch
        trainer.iteration = start_iteration
        trainer.train()
    elif args.cmd == 'test':
        validator = Validator(
            cmd=args.cmd,
            cuda=cuda,
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            log_file=log_file,
            print_freq=1,
        )
        validator.validate()
    elif args.cmd == 'extract':
        extractor = Extractor(
            cuda=cuda,
            model=model,
            val_loader=val_loader,
            log_file=log_file,
            feature_dir=args.feature_dir,
            flatten_feature=True,
            print_freq=1,
        )
        extractor.extract()


if __name__ == '__main__':
    main()
