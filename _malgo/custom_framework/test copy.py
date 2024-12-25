import os.path as osp
from mmengine.dataset import Compose

import torch
import copy

from mmaction.registry import DATASETS
from mmaction.registry import MODELS
from mmaction.registry import METRICS
from mmengine.runner import Runner

# from mmaction.datasets.transforms.loading import (
#     DecordInit, 
#     SampleFrames, 
#     DecordDecode,
# )
# from mmaction.datasets.transforms.processing import (
#     Resize,
#     MultiScaleCrop,
#     Flip,
# )

# from mmaction.datasets.transforms.formatting import (
#     FormatShape,
#     PackActionInputs,
# )

from mmaction.utils import register_all_modules
register_all_modules(init_default_scope=True)

ann_file_train = 'data/malgo/malgo_train_video.txt'
ann_file_val = 'data/malgo/malgo_val_video.txt'
data_root_train = 'data/malgo/train'
data_root_val = 'data/malgo/val'
dataset_type = 'VideoDataset'

train_pipeline_cfg = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=8, frame_interval=1, num_clips=4, type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(-1,256,), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]

val_pipeline_cfg = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=8, frame_interval=1, num_clips=4, test_mode = True, type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(-1,256,), type='Resize'),
    dict(keep_ratio=False, scale=(224,224,), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]

train_dataset_cfg = dict(
    ann_file=ann_file_train,
    data_prefix=dict(video=data_root_train),
    pipeline=train_pipeline_cfg,
    type='VideoDataset'
)

val_dataset_cfg = dict(
    ann_file=ann_file_val,
    data_prefix=dict(video=data_root_val),
    pipeline=val_pipeline_cfg,
    type='VideoDataset'
)

train_dataset = DATASETS.build(train_dataset_cfg)

packed_results = train_dataset[0]

inputs = packed_results['inputs']
data_sample = packed_results['data_samples']

# TODO put this on demo
# print('shape of the inputs: ', inputs.shape)
# print('data sample: ', data_sample)

# Can change num_workers to 1. But need to put this on main

BATCH_SIZE = 2

train_dataloader_cfg = dict(
    batch_size=BATCH_SIZE,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset_cfg)

val_dataloader_cfg = dict(
    batch_size=BATCH_SIZE,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset_cfg)

train_data_loader = Runner.build_dataloader(dataloader=train_dataloader_cfg)
val_data_loader = Runner.build_dataloader(dataloader=val_dataloader_cfg)

batched_packed_results = next(iter(train_data_loader))

batched_inputs = batched_packed_results['inputs']
batched_data_sample = batched_packed_results['data_samples']

assert len(batched_inputs) == BATCH_SIZE
assert len(batched_data_sample) == BATCH_SIZE


data_preprocessor_cfg = dict(
    type='ActionDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    format_shape='NCHW')

# TODO put this on demo
# data_preprocessor = MODELS.build(data_preprocessor_cfg)

# preprocessed_inputs = data_preprocessor(batched_packed_results)
# print(preprocessed_inputs['inputs'].shape)

model_cfg = dict(
    type='Recognizer2D',
backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False
    ),
    cls_head=dict(
        type='TSNHead',
        num_classes=4,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'
    ),
    data_preprocessor = data_preprocessor_cfg)

model = MODELS.build(model_cfg)

# TODO put this on demo
# Train
model.train()
model.init_weights()
data_batch_train = copy.deepcopy(batched_packed_results)
data = model.data_preprocessor(data_batch_train, training=True)
loss = model(**data, mode='loss')
print('loss dict: ', loss)

# Test
with torch.no_grad():
    model.eval()
    data_batch_test = copy.deepcopy(batched_packed_results)
    data = model.data_preprocessor(data_batch_test, training=False)
    predictions = model(**data, mode='predict')
print('Label of Sample[0]', predictions[0].gt_label)
print('Scores of Sample[0]', predictions[0].pred_score)


metric_cfg = dict(type='AccMetric')

metric = METRICS.build(metric_cfg)

data_samples = [d.to_dict() for d in predictions]

metric.process(batched_packed_results, data_samples)
acc = metric.compute_metrics(metric.results)
print(acc)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
val_cfg = dict(type='ValLoop')

optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.01))

runner = Runner(model=model_cfg, work_dir='./work_dirs/test_last',
                train_dataloader=train_dataloader_cfg,
                train_cfg=train_cfg,
                val_dataloader=val_dataloader_cfg,
                val_cfg=val_cfg,
                optim_wrapper=optim_wrapper,
                val_evaluator=[metric_cfg],
                default_scope='mmaction')
runner.train()