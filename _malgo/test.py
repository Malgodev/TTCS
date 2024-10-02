import os.path as osp
from mmengine.dataset import Compose
from _malgo.custom_pipeline import VideoInit, VideoSample, VideoDecode, VideoResize, VideoCrop, VideoFormat, VideoPack

pipeline_cfg = [
    dict(type='VideoInit'),
    dict(type='VideoSample', clip_len=16, num_clips=1, test_mode=False),
    dict(type='VideoDecode'),
    dict(type='VideoResize', r_size=256),
    dict(type='VideoCrop', c_size=224),
    dict(type='VideoFormat'),
    dict(type='VideoPack')
]

pipeline = Compose(pipeline_cfg)
data_prefix = 'data/kinetics400_tiny/train'
results = dict(filename=osp.join(data_prefix, 'D32_1gwq35E.mp4'), label=0)
packed_results = pipeline(results)

inputs = packed_results['inputs']
data_sample = packed_results['data_samples']

print('shape of the inputs: ', inputs.shape)

# Get metainfo of the inputs
print('image_shape: ', data_sample.img_shape)
print('num_clips: ', data_sample.num_clips)
print('clip_len: ', data_sample.clip_len)

# Get label of the inputs
print('label: ', data_sample.gt_label)