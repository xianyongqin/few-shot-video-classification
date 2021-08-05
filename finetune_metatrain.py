import os
import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from utils import Logger
from train import train_epoch
from validation import val_epoch
from datasets.kinetics import Kinetics
from datasets.something import Something
from datasets.ucf101 import UCF101

if __name__ == '__main__':
    opt = parse_opts()

    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)


    torch.backends.cudnn.benchmark = True

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
   # print(model)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    # if not opt.no_cuda:
    criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        if opt.dataset == 'kinetics':
            training_data = Kinetics(
                opt.video_path,
                opt.train_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_samples_for_each_video)

        elif opt.dataset == 'something':
            training_data = Something(
                opt.video_path,
                opt.train_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_samples_for_each_video)

        elif opt.dataset == 'ucf101':
            training_data = UCF101(
                opt.video_path,
                opt.train_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_samples_for_each_video)


        print(len(training_data))

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)


        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        if opt.adam:
            optimizer = optim.Adam(parameters, lr=opt.learning_rate, betas=(0.5, 0.999))
        else:
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)
        # scheduler = lr_scheduler.ReduceLROnPlateau(
        #    optimizer, 'min', patience=opt.lr_patience)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        if opt.dataset == 'kinetics':
            validation_data = Kinetics(
                opt.val_video_path,
                opt.val_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_val_samples)
        elif opt.dataset == 'something':
            validation_data = Something(
                opt.val_video_path,
                opt.val_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_val_samples)

        elif opt.dataset == 'ucf101':
            validation_data = UCF101(
                opt.val_video_path,
                opt.val_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_val_samples)


        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

        print('# of validation clips %d' % len(validation_data))

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger)

        if not opt.no_val:
            if i % opt.val_every == 0:
                validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)

        scheduler.step()

