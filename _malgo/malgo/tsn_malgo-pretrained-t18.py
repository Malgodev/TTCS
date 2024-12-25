ann_file_train = 'data/malgo/malgo_train_video.txt'
ann_file_val = 'data/malgo/malgo_val_video.txt'
auto_scale_lr = dict(base_batch_size=256, enable=False)
data_root_train = 'data/malgo/malgo_train'
data_root_val = 'data/malgo/malgo_val'
dataset_type = 'VideoDataset'


default_hooks = dict(
    checkpoint=dict(
        interval=3, max_keep_ckpts=3, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))

default_scope = 'mmaction' #

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

file_client_args = dict(io_backend='disk')
load_from = None # sẽ thêm ở đây
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',  # Sử dụng ResNet
        pretrained='torchvision://resnet50',  # Sử dụng trọng số đã được huấn luyện từ ResNet-18
        depth=50,  # Độ sâu của ResNet
        norm_eval=False
    ),
    cls_head=dict(
        type='TSNHead',  # Model sử dụng
        num_classes=4,  # Số lượng action classes (tấn công, đỡ, nhảy, ...)
        in_channels=2048,  # Số lượng channels, tương ứng với depth = 18
        spatial_type='avg',  # Lấy trung bình pooling
        consensus=dict(type='AvgConsensus', dim=1),  # Cấu hình của Consensus module
        dropout_ratio=0.5,  # Tỉ lệ bỏ một số nơ-ron trong quá trình train, tránh overfitting
        init_std=0.01,  # Standard deviation khởi tạo, có trọng số là 0.01
        average_clips='prob'  # Trung bình có trọng số theo xác suất
    ),
    data_preprocessor=dict(  # Tiền xử lý dữ liệu
        type='ActionDataPreprocessor',  # Loại
        mean=[123.675, 116.28, 103.53],  # Giá trị trung bình của RGB chuẩn hóa dữ liệu đầu vào
        std=[58.395, 57.12, 57.375],  # Độ lệch chuẩn của RGB để chuẩn hóa
        format_shape='NCHW'  # Định dạng sau xử lý trong đó
        # N: Batch size
        # C: Channels (RGB in this case)
        # H: Height
        # W: Width
    ),
    test_cfg=None,
    train_cfg=None)

optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001))

param_scheduler = [
    dict(
        begin=0, 
        by_epoch=True, 
        end=10,
        gamma=0.1, 
        milestones=[ 
            4,
            8,
        ],
        type='MultiStepLR'),
]

resume = False  # Và ở đây

# Phần dưới này là test hiện đang để chung với val chưa có đủ dataset
BATCH_SIZE = 2

test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=BATCH_SIZE,
    dataset=dict(
        ann_file='data/malgo/malgo_val_video.txt',
        data_prefix=dict(video='data/malgo/val'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=1,
                frame_interval=1,
                num_clips=25,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='TenCrop'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='TenCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
train_dataloader = dict(
    batch_size=BATCH_SIZE,
    dataset=dict(
        ann_file='data/malgo/malgo_train_video.txt',
        data_prefix=dict(video='data/malgo/train'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=1, frame_interval=1, num_clips=8,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(
                input_size=224,
                max_wh_scale_gap=1,
                random_crop=False,
                scales=(
                    1,
                    0.875,
                    0.75,
                    0.66,
                ),
                type='MultiScaleCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='VideoDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=1, frame_interval=1, num_clips=8, type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(
        input_size=224,
        max_wh_scale_gap=1,
        random_crop=False,
        scales=(
            1,
            0.875,
            0.75,
            0.66,
        ),
        type='MultiScaleCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]

val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=BATCH_SIZE,
    dataset=dict(
        ann_file='data/malgo/malgo_val_video.txt',
        data_prefix=dict(video='data/malgo/val'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=1,
                frame_interval=1,
                num_clips=8,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]

visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])