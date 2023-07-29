freq_file_in_loss = 'mllt/appendix/chestdevkit_part_0_part_0_part_0_part_1_part_1/class_freq.pkl'
# work_dir = './work_dirs/LT_xchest_resnet50_pfc_DB_part3_' + 'sampling4'
work_dir = './work_dirs/LT_xchest_resnet50_pfc_DB_' + 'part_0_part_0_part_0_part_1_part_1'
ann_csv_train = "/label/part_0_part_0_part_0_part_1_part_1.csv"
ann_csv_val =   "/label/part_0_part_0_part_0_part_1_part_1.csv"
ann_csv_test = '/label/development_with_size.csv'
# type='MLP',
# in_channels=2048,
# bottle_neck = 1024,
# out_channels=512,
# dropout1=0.5,
# dropout2=0.5),
# model settings
model = dict(
    type='SimpleClassifier',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='PFC',
        in_channels=2048,
        out_channels=256,
        dropout=0.5),
    head=dict(
        type='ClsHead',
        in_channels=256,
        num_classes=26,
        method='fc',
        loss_cls=dict(
            type='ResampleLoss', use_sigmoid=True,
            reweight_func='rebalance',
            focal=dict(focal=True, balance_param=2.0, gamma=2),
            logit_reg=dict(neg_scale=2.0, init_bias=0.05),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
            loss_weight=1.0, freq_file=freq_file_in_loss)))
# model training and testing settings
train_cfg = dict()
test_cfg = dict()

# dataset settings
dataset_type = 'XChestDataset2'
data_root = '/content/drive/MyDrive/official_data_iccv_2'
online_data_root = None
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
extra_aug = dict(
    photo_metric_distortion=dict(
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    random_crop=dict(
        min_crop_size=0.8
    )
)

img_size=512
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=2,
    sampler='ClassAware',
    train=dict(
            type=dataset_type,
            ann_file=data_root + ann_csv_train,
            LT_ann_file = None,
            img_prefix=data_root,
            img_scale=(img_size, img_size),
            img_norm_cfg=img_norm_cfg,
            extra_aug=extra_aug,
            size_divisor=32,
            resize_keep_ratio=False,
            flip_ratio=0.5),
    val=dict(
        type=dataset_type,
        ann_file=data_root + ann_csv_val,
        img_prefix=data_root,
        img_scale=(img_size, img_size),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        resize_keep_ratio=False,
        flip_ratio=0),
    test=dict(
            type=dataset_type,
            ann_file=data_root + ann_csv_test,
            # class_split=online_data_root + 'coco/longtail2017/class_split.pkl',
            img_prefix=data_root + '/test',
            img_scale=(img_size, img_size),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32, 
            resize_keep_ratio=False,
            flip_ratio=0))
    # val=dict(
    #     type=dataset_type,
    #     ann_file=data_root + '/label/part_1.csv',
    #     img_prefix=data_root,
    #     img_scale=(img_size, img_size),
    #     img_norm_cfg=img_norm_cfg,
    #     size_divisor=32,
    #     resize_keep_ratio=False,
    #     flip_ratio=0))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[5,7])  # 8: [5,7]) 4: [2,3]) 40: [25,35]) 80: [55,75])
checkpoint_config = dict(interval=8)
# yapf:disable
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
evaluation = dict(interval=5)
# runtime settings
start_epoch=0
total_epochs = 5
dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = None
if start_epoch > 0:
    resume_from = work_dir + '/epoch_{}.pth'.format(start_epoch)
    print("start from epoch {}".format(start_epoch))
else:
    resume_from = None
workflow = [('train', 1)]