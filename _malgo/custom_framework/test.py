# TODO this will later put in tsn_r18_(***}

import os.path as osp
from mmengine.dataset import Compose
from mmengine.registry import DATASETS

from pipeline import (
    VideoInit,
    VideoSample,
    VideoDecode,
    VideoResize,
    VideoCrop,
    VideoFormat,
    VideoPack)
from dataset import DatasetMalgo
from data_preprocessor import (
    DataPreprocessorMalgo)
from recognizer import (
    BackBoneMalgo,
    ClsHeadMalgo,
    RecognizerMalgo
)
from metric import (
    AccuracyMetric
)

from mmaction.utils import register_all_modules

register_all_modules(init_default_scope=True)

train_pipeline_cfg = [
    dict(type='VideoInit'),
    dict(type='VideoSample', clip_len=16, num_clips=1, test_mode=False),
    dict(type='VideoDecode'),
    dict(type='VideoResize', r_size=256),
    dict(type='VideoCrop', c_size=224),
    dict(type='VideoFormat'),
    dict(type='VideoPack')
]

val_pipeline_cfg = [
    dict(type='VideoInit'),
    dict(type='VideoSample', clip_len=16, num_clips=5, test_mode=True),
    dict(type='VideoDecode'),
    dict(type='VideoResize', r_size=256),
    dict(type='VideoCrop', c_size=224),
    dict(type='VideoFormat'),
    dict(type='VideoPack')
]

train_dataset_cfg = dict(
    type='DatasetMalgo',
    ann_file='malgo_train_video.txt',
    pipeline=train_pipeline_cfg,
    data_root='data/malgo/',
    data_prefix=dict(video='train'))

val_dataset_cfg = dict(
    type='DatasetMalgo',
    ann_file='malgo_val_video.txt',
    pipeline=val_pipeline_cfg,
    data_root='data/malgo/',
    data_prefix=dict(video='val'))

train_dataset = DATASETS.build(train_dataset_cfg)

packed_results = train_dataset[0]

inputs = packed_results['inputs']
data_sample = packed_results['data_samples']

from mmengine.runner import Runner

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

from mmaction.registry import MODELS

data_preprocessor_cfg = dict(
    type='DataPreprocessorMalgo',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375])

data_preprocessor = MODELS.build(data_preprocessor_cfg)

preprocessed_inputs = data_preprocessor(batched_packed_results)

import torch
import copy

model_cfg = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50', 
        depth=50,
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
    data_preprocessor = dict(
        type='DataPreprocessorMalgo',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))

model = MODELS.build(model_cfg)

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
print('Scores of Sample[0]', predictions[0].pred_scores)

from mmaction.registry import METRICS

metric_cfg = dict(type='AccuracyMetric', topk=(1, 5))

metric = METRICS.build(metric_cfg)

data_samples = [d.to_dict() for d in predictions]

metric.process(batched_packed_results, data_samples)
acc = metric.compute_metrics(metric.results)
print(acc)

from mmengine.runner import Runner

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
val_cfg = dict(type='ValLoop')

optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.01))

runner = Runner(model=model_cfg, work_dir='./work_dirs/tmp',
                train_dataloader=train_dataloader_cfg,
                train_cfg=train_cfg,
                val_dataloader=val_dataloader_cfg,
                val_cfg=val_cfg,
                optim_wrapper=optim_wrapper,
                val_evaluator=[metric_cfg],
                default_scope='mmaction')
runner.train()