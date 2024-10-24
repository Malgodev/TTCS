model = dict(
    type='Recognizer2D', # Xác định hình ảnh sử dụng 2D
    backbone=dict( 
        type='ResNet', # Sử dụng Resnet
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead', # Model sử dụng
        num_classess=11, # Số lượng action classess (tấn công, đỡ, nhảy, ...)
        in_channels=2048, # Số lượng channels, tương ứng với depth = 50
        spatial_type='avg', # Lấy trung bình pooling
        consensus=dict(type='AvgConsensus', dim=1), # Cấu hình của Consensus module
        dropout_ratio=0.4, # Tỉ lệ bỏ một số noron trong qua trình train, tránh overffiting
        init_std=0.01, # standard deviation khởi tạo, có trọng số là 0.01
        average_clips='prob'), # trung bình có trọng số theo xác xuất
    data_preprocessor=dict(  # Tiền xử lý dữ liệu
        type='ActionDataPreprocessor',  # loại
        mean=[123.675, 116.28, 103.53],  # giá trị trung bình của RGB chuẩn hoá dữ liệu đầu vào
        std=[58.395, 57.12, 57.375],  # độ lệch chuẩn của RGB để chuẩn hoá.
        format_shape='NCHW'), # định dạng sau xử lý trong đó
        # N: Batch size
        # C: Channels (RGB in this case)
        # H: Height
        # W: Width
    train_cfg=None,
    test_cfg=None
)

# python demo/webcam_demo.py 
# demo/demo_configs/i3d_r50_32x2x1_video_infer.py 
# checkpoints/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_20220812-e213c223.pth 
# tools/data/kinetics/label_map_k400.txt 
# --average-size 5 
# --threshold 0.2

# python demo/webcam_demo.py 
# demo/demo_configs/tsn_r50_1x1x8_video_infer.py
# https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth
# tools/data/kinetics/label_map_k400.txt 
# --average-size 5 
# --threshold 0.2 

# This configuration file outlines the setup for a 2D action recognition model in MMAction, a toolbox for video understanding tasks like action recognition. Here's a breakdown of key components and their purposes:

# Model Configuration
# type='Recognizer2D': The model is a 2D action recognizer, which means it processes 2D image frames to predict actions.
# backbone: Uses a ResNet-50 model (depth=50) pretrained on ImageNet for feature extraction.
# cls_head (TSNHead): Implements a Temporal Segment Network (TSN) head for classifying actions into 400 categories (num_classes=400) with a dropout ratio of 0.4.
# data_preprocessor: Normalizes input frames with specific mean and standard deviation values and formats them in NCHW format (channels first).
# Dataset and Pipeline
# dataset_type='RawframeDataset': The model is trained on a dataset where videos are pre-extracted into raw frames.
# data_root & ann_file: Specifies the path to the raw frames and corresponding annotations for training, validation, and testing.
# train/val/test_pipeline: Defines how the frames are sampled, processed (resized, cropped, flipped), and formatted during training, validation, and testing. For instance, in training, frames are cropped, resized to 224x224, and augmented with random flips.
# sampler: Specifies how frames are sampled during training and validation, with shuffling enabled for training but disabled for validation and testing.
# Training, Validation, and Testing
# train_cfg: The model will be trained for 100 epochs, and validation will begin after the first epoch with a validation interval of 1 epoch.
# val_cfg/test_cfg: Specifies that the validation and testing loops will use ValLoop and TestLoop, respectively.
# Optimization and Scheduler
# optim_wrapper: Uses SGD with momentum (0.9) and a learning rate (lr=0.01). Gradient clipping is applied with a max norm of 40.
# param_scheduler: A learning rate scheduler based on MultiStepLR, reducing the learning rate at epochs 40 and 80 by a factor of 0.1.
# Runtime and Hooks
# default_hooks: Defines hooks for logging, checkpointing (saving the model every 3 epochs, keeping the best one), and synchronizing buffers during distributed training.
# Differences between Action Localization, Spatio-Temporal Action Detection, and Action Recognition:
# Action Localization: Focuses on identifying not only which action is happening but also where it occurs spatially in the video frames. It typically requires bounding box annotations to localize actions.
# Spatio-Temporal Action Detection: Combines both spatial and temporal dimensions, detecting actions across space and time, usually within untrimmed videos. It involves detecting when and where an action occurs throughout a video.
# Action Recognition: This task focuses purely on classifying which action is being performed in a video or a set of frames, without concern for localizing the action in space or time. The model in your config is for action recognition, classifying the action based on entire video clips.