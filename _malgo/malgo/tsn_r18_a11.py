model = dict(
    type='Recognizer2D',  # Xác định hình ảnh sử dụng 2D
    backbone=dict(
        type='ResNet',  # Sử dụng ResNet
        pretrained='torchvision://resnet18',  # Sử dụng trọng số đã được huấn luyện từ ResNet-18
        depth=18,  # Độ sâu của ResNet
        norm_eval=False  # Không đặt các lớp BN ở chế độ đánh giá
    ),
    cls_head=dict(
        type='TSNHead',  # Model sử dụng
        num_classes=11,  # Số lượng action classes (tấn công, đỡ, nhảy, ...)
        in_channels=512,  # Số lượng channels, tương ứng với depth = 18
        spatial_type='avg',  # Lấy trung bình pooling
        consensus=dict(type='AvgConsensus', dim=1),  # Cấu hình của Consensus module
        dropout_ratio=0.4,  # Tỉ lệ bỏ một số nơ-ron trong quá trình train, tránh overfitting
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
    train_cfg=None,  # Cấu hình huấn luyện
    test_cfg=None  # Cấu hình kiểm tra
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