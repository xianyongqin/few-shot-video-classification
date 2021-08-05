import os
import torch
from torch import nn
import torch.nn.parallel

from opts import parse_opts
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from datasets.kinetics_episode import make_video_names, KineticsVideoList
from datasets.something_episode import SomethingVideoList, make_something_video_names
from datasets.ucf101_episode import UCFVideoList, make_ucf_video_names

from utils import setup_logger, AverageMeter, count_acc, euclidean_metric
import time
from batch_sampler import CategoriesSampler
import torch.optim as optim
from clip_model import generate_model

import numpy as np
import pickle

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


class CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(CLASSIFIER, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

def get_classifier_weights(embedding_shot, target_shot, lr=0.01, nepoch=5):
    classifier = CLASSIFIER(embedding_shot.size(1), opt.test_way)
    classifier.apply(weights_init)
    criterion = nn.NLLLoss()

    classifier.cuda()
    criterion.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.5, 0.999))
    for i in range(nepoch):
        optimizer.zero_grad()
        output = classifier(embedding_shot)
        loss = criterion(output, target_shot)
        #print(loss.data)
        loss.backward()
        optimizer.step()

    return classifier.fc.weight.data


def train_epoch(support_data_loader, model, classifier, criterion, optimizer):
    classifier.train()
    support_clip_embedding = torch.FloatTensor(opt.test_novel_way*opt.shot*opt.n_samples_for_each_video, opt.emb_dim).cuda()
    support_clip_label = torch.LongTensor(opt.test_novel_way*opt.shot*opt.n_samples_for_each_video).cuda()
    batch_size = opt.n_samples_for_each_video
    with torch.no_grad():
        cur_loc = 0
        for i, (data, label) in enumerate(support_data_loader):
            batch_embedding = model(data.cuda())
            cur_batch = batch_embedding.size(0)
            support_clip_embedding[cur_loc:cur_loc+cur_batch] = batch_embedding.squeeze()
            support_clip_label[cur_loc:cur_loc+cur_batch] = label.cuda()
            cur_loc += cur_batch

    optimizer.zero_grad()
    output = classifier(support_clip_embedding)
    loss = criterion(output, support_clip_label)
    loss.backward()
    optimizer.step()


def val_epoch(query_novel_data_loader, query_base_data_loader, model, classifier):
    classifier.eval()
    query_novel_clip_embedding = torch.FloatTensor(opt.test_novel_way * opt.query_novel * opt.n_val_samples, opt.emb_dim).cuda()
    query_base_clip_embedding = torch.FloatTensor(opt.test_base_way * opt.query_base * opt.n_val_samples, opt.emb_dim).cuda()

    with torch.no_grad():
        cur_loc = 0
        for i, (data, label) in enumerate(query_novel_data_loader):
            batch_embedding = model(data.cuda())
            cur_batch = batch_embedding.size(0)
            query_novel_clip_embedding[cur_loc:cur_loc+cur_batch] = batch_embedding.squeeze()
            cur_loc += cur_batch

        cur_loc = 0
        for i, (data, label) in enumerate(query_base_data_loader):
            batch_embedding = model(data.cuda())
            cur_batch = batch_embedding.size(0)
            query_base_clip_embedding[cur_loc:cur_loc+cur_batch] = batch_embedding.squeeze()
            cur_loc += cur_batch


        base_clip_logits = torch.exp(classifier(query_base_clip_embedding))
        base_logits = base_clip_logits.reshape(opt.query_base * opt.test_base_way, opt.n_val_samples, -1).mean(dim=1)

        novel_clip_logits = torch.exp(classifier(query_novel_clip_embedding))
        novel_logits = novel_clip_logits.reshape(opt.query_novel * opt.test_novel_way, opt.n_val_samples, -1).mean(dim=1)

        all_logits = torch.cat((base_logits, novel_logits))

        query_base_labels = torch.arange(opt.test_base_way).repeat(opt.query_base)
        query_novel_labels = torch.arange(opt.test_novel_way).repeat(opt.query_novel) + opt.test_base_way
        query_all_labels = torch.cat((query_base_labels, query_novel_labels)).cuda()

        is_base = torch.arange(opt.test_base_way*opt.query_base)
        is_novel = torch.arange(opt.test_base_way*opt.query_base, opt.test_base_way*opt.query_base+opt.test_novel_way*opt.query_novel)

        predicted_labels = torch.argmax(all_logits, dim=1)
        # print(predicted_labels)
        # print(query_all_labels)
        acc_all = (predicted_labels == query_all_labels).type(torch.cuda.FloatTensor).mean().item()
        acc_base = (predicted_labels[is_base] == query_all_labels[is_base]).type(torch.cuda.FloatTensor).mean().item()
        acc_novel = (predicted_labels[is_novel] == query_all_labels[is_novel]).type(torch.cuda.FloatTensor).mean().item()


    return acc_novel, acc_base, acc_all, predicted_labels[is_novel]


def append_base_weight(classifier, base_weight):

    weight = torch.cat((base_weight, classifier.fc.weight.data))
    classifier.fc = nn.Linear(opt.emb_dim,opt.test_base_way+opt.test_novel_way)
    classifier.fc = classifier.fc.cuda()

    classifier.fc.weight.data = weight

    return classifier


def meta_test_episode(support_data_loader, query_novel_data_loader, query_base_data_loader, model, base_weight, opt):
    model.eval()
    # train classifier
    classifier = CLASSIFIER(opt.emb_dim, opt.test_novel_way)
    classifier.apply(weights_init)
    criterion = nn.NLLLoss()
    classifier.cuda()
    criterion.cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    for i in range(opt.nepoch):
        train_epoch(support_data_loader, model, classifier, criterion, optimizer)
        #acc = val_epoch(query_data_loader, model, classifier)
        #print(acc)

    classifier = append_base_weight(classifier, base_weight)
    acc_novel, acc_base, acc_all, predicted_novel_labels = val_epoch(query_novel_data_loader, query_base_data_loader, model, classifier)
    return acc_novel, acc_base, acc_all, predicted_novel_labels


if __name__ == '__main__':
    opt = parse_opts()
    print(opt)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)

    opt.arch = '{}-{}'.format(opt.clip_model, opt.clip_model_depth)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)

    # Setup logging system
    logger = setup_logger(
        "validation",
        opt.result_path,
        0,
        'results.txt'
    )
    logger.debug(opt)
    print(opt.lr)
    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    torch.backends.cudnn.benchmark = True

    torch.manual_seed(opt.manual_seed)

    model = generate_model(opt)
    if opt.resume_path:
        print('loading pretrained model {}'.format(opt.resume_path))
        pretrain = torch.load(opt.resume_path)
        model.load_state_dict(pretrain['state_dict'])

    base_weight = model.module.fc.weight.data
    model = nn.Sequential(*list(model.module.children())[:-1])
    #print(model)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)


    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            opt.scales, opt.sample_size, crop_positions=['c'])

    train_spatial_transform = Compose([
        crop_method,
        RandomHorizontalFlip(),
        ToTensor(opt.norm_value), norm_method
    ])
    train_temporal_transform = TemporalRandomCrop(opt.sample_duration)

    test_spatial_transform = Compose([
        Scale(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value), norm_method
    ])
    test_temporal_transform = TemporalRandomCrop(opt.sample_duration)

    if opt.dataset == 'kinetics100':
        test_novel_videos, test_novel_labels = make_video_names(opt.test_novel_video_path, opt.test_novel_list_path)
        test_base_videos, test_base_labels = make_video_names(opt.test_base_video_path, opt.test_base_list_path)

    elif opt.dataset == 'something':
        test_novel_videos, test_novel_labels = make_something_video_names(opt.test_novel_video_path, opt.test_novel_list_path)
        test_base_videos, test_base_labels = make_something_video_names(opt.test_base_video_path, opt.test_base_list_path)

    elif opt.dataset == 'ucf101':
        test_novel_videos, test_novel_labels = make_ucf_video_names(opt.test_novel_video_path, opt.test_novel_list_path)
        test_base_videos, test_base_labels = make_ucf_video_names(opt.test_base_video_path, opt.test_base_list_path)


    episode_novel_sampler = CategoriesSampler(test_novel_labels,
                                opt.nepisode, opt.test_novel_way, opt.shot + opt.query_novel)

    episode_base_sampler = CategoriesSampler(test_base_labels,
                                opt.nepisode, opt.test_base_way, opt.query_base)


    episode_time = AverageMeter()
    accuracies_base = AverageMeter()
    accuracies_novel = AverageMeter()
    accuracies_all = AverageMeter()


    n_classes = len(np.unique(test_base_labels)) + len(np.unique(test_novel_labels))

    confusion_matrix = torch.zeros(n_classes, n_classes)


    for i, (batch_novel_idx, batch_base_idx) in enumerate(zip(episode_novel_sampler, episode_base_sampler)):
        k = opt.test_novel_way * opt.shot
        support_videos = [test_novel_videos[j] for j in batch_novel_idx[:k]]
        support_labels = torch.arange(opt.test_novel_way).repeat(opt.shot)

        query_novel_videos = [test_novel_videos[j] for j in batch_novel_idx[k:]]
        query_novel_labels = torch.arange(opt.test_novel_way).repeat(opt.query_novel)

        query_base_videos = [test_base_videos[j] for j in batch_base_idx]
        query_base_labels = torch.arange(opt.test_base_way).repeat(opt.query_base)

        cur_classes = list(np.arange(opt.test_base_way))
        for j in batch_novel_idx[:opt.test_novel_way]:
            cur_classes.append(test_novel_labels[j]+len(np.unique(test_base_labels)))

        if opt.dataset == 'kinetics100':
            support_data_loader = torch.utils.data.DataLoader(
                KineticsVideoList(
                    support_videos,
                    support_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_samples_for_each_video),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            query_novel_data_loader = torch.utils.data.DataLoader(
                KineticsVideoList(
                    query_novel_videos,
                    query_novel_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_val_samples),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            query_base_data_loader = torch.utils.data.DataLoader(
                KineticsVideoList(
                    query_base_videos,
                    query_base_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_val_samples),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)


        elif opt.dataset == 'something':
            support_data_loader = torch.utils.data.DataLoader(
                SomethingVideoList(
                    support_videos,
                    support_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_samples_for_each_video),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            query_novel_data_loader = torch.utils.data.DataLoader(
                SomethingVideoList(
                    query_novel_videos,
                    query_novel_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_val_samples),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            query_base_data_loader = torch.utils.data.DataLoader(
                SomethingVideoList(
                    query_base_videos,
                    query_base_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_val_samples),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

        elif opt.dataset == 'ucf101':
            support_data_loader = torch.utils.data.DataLoader(
                UCFVideoList(
                    support_videos,
                    support_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_samples_for_each_video),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            query_novel_data_loader = torch.utils.data.DataLoader(
                UCFVideoList(
                    query_novel_videos,
                    query_novel_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_val_samples),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            query_base_data_loader = torch.utils.data.DataLoader(
                UCFVideoList(
                    query_base_videos,
                    query_base_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_val_samples),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

        end_time = time.time()
        acc_novel, acc_base, acc_all, predicted_novel_labels = meta_test_episode(support_data_loader, query_novel_data_loader, query_base_data_loader, model, base_weight,  opt)
        accuracies_novel.update(acc_novel)
        accuracies_base.update(acc_base)
        accuracies_all.update(acc_all)

        episode_time.update(time.time() - end_time)

        logger.info('Episode: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Acc_novel {acc_novel.val:.3f} ({acc_novel.avg:.3f})\t'
              'Acc_base {acc_base.val:.3f} ({acc_base.avg:.3f})\t'
              'Acc_all {acc_all.val:.3f} ({acc_all.avg:.3f})\t'.format(
                  i + 1,
                  opt.nepisode,
                  batch_time=episode_time,
                  acc_novel=accuracies_novel,
                  acc_base=accuracies_base,
                  acc_all=accuracies_all))


        #
        # for t, p in zip(query_novel_labels.view(-1), predicted_novel_labels.view(-1)):
        #     confusion_matrix[cur_classes[t.long()+opt.test_base_way], cur_classes[p.long()]] += 1
        #
        #
        # if i % 10 == 0:
        #     confusion_matrix_save = confusion_matrix.numpy()
        #     with open(os.path.join(opt.result_path, 'conf_mat_' + str(i) + '.pkl'), 'wb') as fp:
        #         pickle.dump(confusion_matrix_save, fp)