model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False
    ),
    cls_head=dict(
        type='TSNHead',
        num_classess=11,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
    )
)

# python demo/webcam_demo.py 
# demo/demo_configs/i3d_r50_32x2x1_video_infer.py 
# checkpoints/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_20220812-e213c223.pth 
# tools/data/kinetics/label_map_k400.txt 
# --average-size 5 
# --threshold 0.2