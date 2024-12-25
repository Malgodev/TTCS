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