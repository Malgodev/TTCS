import os.path as osp
from mmaction.utils import register_all_modules
register_all_modules(init_default_scope=True)


# pipe line modules
from pipeline import VideoInit, VideoSample, VideoDecode, VideoResize, VideoCrop, VideoFormat, VideoPack

# dataset modules
from mmaction.registry import DATASETS
from dataset import DatasetZelda


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
    type='DatasetZelda',
    ann_file='kinetics_tiny_train_video.txt',
    pipeline=train_pipeline_cfg,
    data_root='data/kinetics400_tiny/',
    data_prefix=dict(video='train'))

val_dataset_cfg = dict(
    type='DatasetZelda',
    ann_file='kinetics_tiny_val_video.txt',
    pipeline=val_pipeline_cfg,
    data_root='data/kinetics400_tiny/',
    data_prefix=dict(video='val'))

train_dataset = DATASETS.build(train_dataset_cfg)

packed_results = train_dataset[0]

inputs = packed_results['inputs']
data_sample = packed_results['data_samples']

print('shape of the inputs: ', inputs.shape)

# Get metainfo of the inputs
print('image_shape: ', data_sample.img_shape)
print('num_clips: ', data_sample.num_clips)
print('clip_len: ', data_sample.clip_len)

# Get label of the inputs
print('label: ', data_sample.gt_label)

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
from data_preprocessor import DataPreprocessorZelda

data_preprocessor_cfg = dict(
    type='DataPreprocessorZelda',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375])

data_preprocessor = MODELS.build(data_preprocessor_cfg)

preprocessed_inputs = data_preprocessor(batched_packed_results)
print(preprocessed_inputs['inputs'].shape)

import torch
import copy
from recognizer import RecognizerZelda

model_cfg = dict(
    type='RecognizerZelda',
    backbone=dict(type='BackBoneZelda'),
    cls_head=dict(
        type='ClsHeadZelda',
        num_classes=2,
        in_channels=128,
        average_clips='prob'),
    data_preprocessor = dict(
        type='DataPreprocessorZelda',
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
from metric import AccuracyMetric

metric_cfg = dict(type='AccuracyMetric', topk=(1, 5))

metric = METRICS.build(metric_cfg)

data_samples = [d.to_dict() for d in predictions]

metric.process(batched_packed_results, data_samples)
acc = metric.compute_metrics(metric.results)
print(acc)




# Test with native pytorch
# import torch.optim as optim
# from mmengine import track_iter_progress


# device = 'cuda' # or 'cpu'
# max_epochs = 10

# optimizer = optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(max_epochs):
#     model.train()
#     losses = []

#     print(type(track_iter_progress(train_data_loader)), type(train_data_loader))

#     for data_batch in train_data_loader:
#         data = model.data_preprocessor(data_batch, training=True)
#         loss_dict = model(**data, mode='loss')
#         loss = loss_dict['loss_cls']

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         losses.append(loss.item())

#     print(f'Epoch[{epoch}]: loss ', sum(losses) / len(train_data_loader))

#     with torch.no_grad():
#         model.eval()
#         for data_batch in val_data_loader:
#             data = model.data_preprocessor(data_batch, training=False)
#             predictions = model(**data, mode='predict')
#             data_samples = [d.to_dict() for d in predictions]
#             metric.process(data_batch, data_samples)

#         acc = metric.acc = metric.compute_metrics(metric.results)
#         for name, topk in acc.items():
#             print(f'{name}: ', topk)

# 


# Test with MMEngine
from mmengine.runner import Runner

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
val_cfg = dict(type='ValLoop')

optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.01))

runner = Runner(model=model_cfg, work_dir='./work_dirs/guide',
                train_dataloader=train_dataloader_cfg,
                train_cfg=train_cfg,
                val_dataloader=val_dataloader_cfg,
                val_cfg=val_cfg,
                optim_wrapper=optim_wrapper,
                val_evaluator=[metric_cfg],
                default_scope='mmaction')
runner.train()