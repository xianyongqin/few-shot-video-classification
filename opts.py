import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--train_way', type=int, default=30)
    parser.add_argument('--test_way', type=int, default=5)
    parser.add_argument('--nepisode', type=int, default=500)

    parser.add_argument('--query_base', type=int, default=5)
    parser.add_argument('--test_base_way', type=int, default=5)

    parser.add_argument('--query_novel', type=int, default=5)
    parser.add_argument('--test_novel_way', type=int, default=5)


    parser.add_argument(
        '--root_path',
        default='/root/data/ActivityNet',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--video_path',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--train_video_path',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--val_video_path',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--test_video_path',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--test_novel_video_path',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--test_base_video_path',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--train_list_path',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--val_list_path',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--yfcc_root',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--yfcc_video_list',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--yfcc_clips',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--test_list_path',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--test_novel_list_path',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--test_base_list_path',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--split_path',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--annotation_path',
        default='kinetics.json',
        type=str,
        help='Annotation file path')
    parser.add_argument(
        '--result_path',
        default='results',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--subset',
        default='train',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--dataset',
        default='kinetics',
        type=str,
        help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
    parser.add_argument(
        '--n_classes',
        default=400,
        type=int,
        help=
        'Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)'
    )
    parser.add_argument(
        '--n_finetune_classes',
        default=400,
        type=int,
        help=
        'Number of classes for fine-tuning. n_classes is set to the number when pretraining.'
    )
    parser.add_argument(
        '--clip_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--clip_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--emb_dim',
        default=512,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--num_videos_per_batch',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--initial_scale',
        default=1.0,
        type=float,
        help='Initial scale for multiscale cropping')
    parser.add_argument(
        '--n_scales',
        default=5,
        type=int,
        help='Number of scales for multiscale cropping')
    parser.add_argument(
        '--scale_step',
        default=0.84089641525,
        type=float,
        help='Scale step for multiscale cropping')
    parser.add_argument(
        '--train_crop',
        default='corner',
        type=str,
        help=
        'Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  ( fme | corner | center)'
    )
    parser.add_argument(
        '--margin',
        default=1,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', default=1e-2, type=float, help='Weight Decay')
    parser.add_argument(
        '--mean_dataset',
        default='activitynet',
        type=str,
        help=
        'dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument(
        '--no_mean_norm',
        action='store_true',
        help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument(
        '--std_norm',
        action='store_true',
        help='If true, inputs are normalized by standard deviation.')
    parser.add_argument(
        '--norm_value',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.set_defaults(std_norm=False)
    parser.add_argument(
        '--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='Currently only support SGD')
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '--lr',
        default=0.001,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--layer_lr',
        nargs='+',
        help='Reduce learning rate',
        default=None,
        type=float
    )
    parser.add_argument(
        '--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default=1000,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--nepoch',
        default=5,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--num_video_per_batch',
        default=2,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--num_clip_per_video',
        default=16,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--n_samples_for_each_video',
        default=1,
        type=int,
        help='Number of validation samples for each activity')
    parser.add_argument(
        '--gem_p',
        default=1,
        type=int,
        help='Number of validation samples for each activity')
    parser.add_argument(
        '--kshot',
        default=None,
        type=int,
        help='# of examples per class.'
    )
    parser.add_argument(
        '--weighting',
        action='store_true',
        help='If true, inputs are not normalized by mean.')
    parser.add_argument(
        '--n_val_samples',
        default=3,
        type=int,
        help='Number of validation samples for each activity')
    parser.add_argument(
        '--num_clips_per_video',
        default=1,
        type=int,
        help='Number of validation samples for each activity')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--pretrained_clip_model_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--yfcc_feature_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--feature_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--novel_feature_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--base_feature_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--pca_path', default='/BS/YFCC100M/nobackup/r25d_sports_feature/pca_transform_Sports.pkl', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--scaler_path', default='/BS/YFCC100M/nobackup/r25d_sports_feature/std_scaler_Sports.pkl', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--ft_begin_index',
        default=0,
        type=int,
        help='Begin block index of fine-tuning')
    parser.add_argument(
        '--topk',
        default=5,
        type=int,
        help='Begin block index of fine-tuning')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.add_argument(
        '--adam',
        action='store_true',
        help='If true, training is not performed.')
    parser.add_argument(
        '--no_last_fc',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    #parser.set_defaults(no_val=False)
    parser.add_argument(
        '--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument(
        '--scale_in_test',
        default=1.0,
        type=float,
        help='Spatial scale in test')
    parser.add_argument(
        '--crop_position_in_test',
        default='c',
        type=str,
        help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument(
        '--no_softmax_in_test',
        action='store_true',
        help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--n_threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--val_every',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--no_hflip',
        action='store_true',
        help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument(
        '--clip_model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--outfile',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--clip_model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--train_feat',
        default='',
        type=str,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--val_feat',
        default='',
        type=str,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
    '--gpu', default = None, type = int, help = 'GPU id')
    parser.add_argument(
    '--distributed', default = False,)
    parser.add_argument(
    '--model', default = 'resnet', type = str,
    help = '(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth', default = 18, type = int, help = 'Depth of resnet (10 | 18 | 34 | 50 | 101)')


    args = parser.parse_args()

    return args
