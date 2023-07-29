freq_file_in_loss = 'mllt/appendix/chestdevkit_final_1_overview_len_20000/class_freq.pkl'
# work_dir = './work_dirs/LT_xchest_resnet50_pfc_DB_part3_' + 'sampling4'
work_dir = './work_dirs/LT_swinTrans_Transformer_DB_' + 'final_1_overview_len_20000'
ann_csv_train = "/label/final_1_overview_len_20000.csv"
# /content/drive/MyDrive/official_data_iccv_2/label/final_1_overview_len_1000.csv
ann_csv_val =   "/label/final_1_overview_len_20000.csv"
ann_csv_test = '/label/final_1_overview_len_20000.csv'
# type='MLP',
# in_channels=2048,
# bottle_neck = 1024,
# out_channels=512,
# dropout1=0.5,
# dropout2=0.5),
# model settings
model = dict(
    type='Query2LabelClassifier',
    backbone=dict(
        backbone='swin_L_384_22k',
        # backbone = "densenet121-res224-all",
        hidden_dim = 2048, #positional encoding input as well
        position_embedding = "sine",
        num_class = 26,
        img_size = 384,
        pretrained = True,
        fuse_imgs = True),
    neck=dict(
        type='Transformer',
        hidden_dim= 2048,
        img_size = 384,
        position_embedding="sine",
        dropout=0.1,
        nheads=4,
        dim_feedforward=8192,
        enc_layers = 1,
        dec_layers = 2,
        num_class = 26,
        keep_other_self_attn_dec = True,
        keep_first_self_attn_dec = True,
        keep_input_proj = True,
        pre_norm= True),
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
dataset_type ="XChestFuse"
plain_non_dist_train = False
data_root = 'C:/Users/Kayne/Desktop/Thao_workspace/Data'
online_data_root = None
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
extra_aug = dict(
    xchest_normalize = dict(
        maxval=255,
        reshape = False
    ),
    random_crop=dict(
        min_crop_size=0.8
    )
)

img_size=384
custom_collate_for_DataLoader = "XChestFuse"
data = dict(
    imgs_per_gpu=2,
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
            flip_ratio=0),
    # val=dict(
    #     type=dataset_type,
    #     ann_file=data_root + ann_csv_val,
    #     img_prefix=data_root,
    #     img_scale=(img_size, img_size),
    #     img_norm_cfg=img_norm_cfg,
    #     size_divisor=32,
    #     resize_keep_ratio=False,
    #     flip_ratio=0),
    # test=dict(
    #         type=dataset_type,
    #         ann_file=data_root + ann_csv_test,
    #         # class_split=online_data_root + 'coco/longtail2017/class_split.pkl',
    #         img_prefix=data_root + '/test',
    #         img_scale=(img_size, img_size),
    #         img_norm_cfg=img_norm_cfg,
    #         size_divisor=32, 
    #         resize_keep_ratio=False,
    #         flip_ratio=0)
            )

# optimizer
# optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=800,
    warmup_ratio=1.0 / 3,
    step=[25,35])  # 8: [5,7]) 4: [2,3]) 40: [25,35]) 80: [55,75])
checkpoint_config = dict(interval=2)
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
total_epochs = 30
dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = None
if start_epoch > 0:
    resume_from = work_dir + '/epoch_{}.pth'.format(start_epoch)
    print("start from epoch {}".format(start_epoch))
else:
    resume_from = None
workflow = [('train', 1)]