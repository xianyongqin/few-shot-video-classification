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
    support_clip_embedding = torch.FloatTensor(opt.test_way*opt.shot*opt.n_samples_for_each_video, opt.emb_dim).cuda()
    support_clip_label = torch.LongTensor(opt.test_way*opt.shot*opt.n_samples_for_each_video).cuda()
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

def val_epoch(query_data_loader, model, classifier):
    classifier.eval()
    batch_size = opt.batch_size
    query_clip_embedding = torch.FloatTensor(opt.test_way * opt.query * opt.n_val_samples, opt.emb_dim).cuda()
    with torch.no_grad():
        cur_loc = 0
        for i, (data, label) in enumerate(query_data_loader):
            batch_embedding = model(data.cuda())
            cur_batch = batch_embedding.size(0)
            query_clip_embedding[cur_loc:cur_loc+cur_batch] = batch_embedding.squeeze()
            cur_loc += cur_batch

        clip_logits = torch.exp(classifier(query_clip_embedding))
        #print(clip_logits)
        logits = clip_logits.reshape(opt.query * opt.test_way, opt.n_val_samples, -1).mean(dim=1)
        query_labels = torch.arange(opt.test_way).repeat(opt.query).cuda()
        acc, pred = count_acc(logits, query_labels)

    return acc

def meta_test_episode(support_data_loader, query_data_loader, model, opt):
    model.eval()
    # train classifier
    classifier = CLASSIFIER(opt.emb_dim, opt.test_way)
    classifier.apply(weights_init)
    criterion = nn.NLLLoss()
    classifier.cuda()
    criterion.cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    for i in range(opt.nepoch):
        train_epoch(support_data_loader, model, classifier, criterion, optimizer)
        #acc = val_epoch(query_data_loader, model, classifier)
        #print(acc)
    acc = val_epoch(query_data_loader, model, classifier)
    return acc


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


    if opt.dataset == 'kinetics100':
        test_videos, test_labels = make_video_names(opt.test_video_path, opt.test_list_path)
    elif opt.dataset == 'something':
        test_videos, test_labels = make_something_video_names(opt.test_video_path, opt.test_list_path)
    elif opt.dataset == 'ucf101':
        test_videos, test_labels = make_ucf_video_names(opt.test_video_path, opt.test_list_path)


    episode_sampler = CategoriesSampler(test_labels,
                                opt.nepisode, opt.test_way, opt.shot + opt.query)


    model = generate_model(opt)
    if opt.resume_path:
        print('loading pretrained model {}'.format(opt.resume_path))
        pretrain = torch.load(opt.resume_path)
        model.load_state_dict(pretrain['state_dict'])

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

    train_spatial_transform = Compose([
        Scale(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value), norm_method
    ])

    train_temporal_transform = TemporalRandomCrop(opt.sample_duration)

    test_spatial_transform = Compose([
        Scale(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value), norm_method
    ])
    test_temporal_transform = TemporalRandomCrop(opt.sample_duration)




    episode_time = AverageMeter()
    accuracies = AverageMeter()


    for i, batch_idx in enumerate(episode_sampler):
        #print(batch_idx)
        k = opt.test_way * opt.shot
        support_videos = [test_videos[j] for j in batch_idx[:k]]
        support_labels = torch.arange(opt.test_way).repeat(opt.shot)

        query_videos = [test_videos[j] for j in batch_idx[k:]]
        query_labels = torch.arange(opt.test_way).repeat(opt.query)

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

            query_data_loader = torch.utils.data.DataLoader(
                KineticsVideoList(
                    query_videos,
                    query_labels,
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

            query_data_loader = torch.utils.data.DataLoader(
                SomethingVideoList(
                    query_videos,
                    query_labels,
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

            query_data_loader = torch.utils.data.DataLoader(
                UCFVideoList(
                    query_videos,
                    query_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_val_samples),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)



        end_time = time.time()
        acc = meta_test_episode(support_data_loader, query_data_loader, model, opt)
        accuracies.update(acc)
        episode_time.update(time.time() - end_time)

        logger.info('Episode: {0}\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  i + 1,
                  batch_time=episode_time,
                  acc=accuracies))