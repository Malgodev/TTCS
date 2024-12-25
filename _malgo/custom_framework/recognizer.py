import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel, BaseModule, Sequential
from mmengine.structures import LabelData
from mmaction.registry import MODELS


@MODELS.register_module()
class BackBoneMalgo(BaseModule):
    # Note: lý do sử dụng 3d là do video sẽ có thêm chiều thời gian
    """
    Backbone network cho việc xử lý video dưới dạng 3D tensors.
    Xử lý cả thông tin không gian (spatial) và thời gian (temporal) của video.
    Args:
        init_cfg (dict): Cấu hình khởi tạo cho các lớp. Mặc định sử dụng:
            - Kaiming cho Conv3d layers
            - Constant cho BatchNorm3d layers
    """    
    def __init__(self, init_cfg=None):
        if init_cfg is None:
            init_cfg = [dict(type='Kaiming', layer='Conv3d', mode='fan_out', nonlinearity="relu"),
                        dict(type='Constant', layer='BatchNorm3d', val=1, bias=0)]

        super(BackBoneMalgo, self).__init__(init_cfg=init_cfg)

        # Conv3D đầu tiên xử lý đồng thời temporal (3) và spatial dimensions (7x7)
        self.conv1 = Sequential(nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                                          stride=(1, 2, 2), padding=(1, 3, 3)),
                                nn.BatchNorm3d(64), nn.ReLU())
        
        # MaxPool3D chỉ thực hiện trên spatial dimensions (H, W)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                    padding=(0, 1, 1))
        # Conv3D thứ hai giảm kích thước không gian và thời gian
        self.conv = Sequential(nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
                               nn.BatchNorm3d(128), nn.ReLU())

    def forward(self, imgs):
        """
        Forward pass của backbone network
        
        Args:
            imgs (torch.Tensor): Input tensor với shape [batch_size*num_views, 3, T, H, W]
                - batch_size: số lượng videos trong batch
                - num_views: số lượng clips cho mỗi video
                - 3: số channels (RGB)
                - T: số frames trong mỗi clip
                - H, W: chiều cao và rộng của frame
                
        Returns:
            features (torch.Tensor): Output features với shape [batch_size*num_views, 128, T/2, H//8, W//8]
                - 128: số channels sau khi qua các conv layers
                - T/2: temporal dimension sau khi downsample
                - H//8, W//8: spatial dimensions sau khi downsample
        """        
        # imgs: [batch_size*num_views, 3, T, H, W]
        # features: [batch_size*num_views, 128, T/2, H//8, W//8]
        features = self.conv(self.maxpool(self.conv1(imgs)))
        return features


@MODELS.register_module()
class ClsHeadMalgo(BaseModule):
    """
    Classification head cho mô hình nhận dạng video 3D
    
    Args:
        num_classes (int): Số lượng classes cần phân loại
        in_channels (int): Số channels của feature input
        dropout (float): Tỉ lệ dropout, mặc định 0.5
        average_clips (str): Phương pháp tính trung bình các clips ('prob', 'score', None)
        init_cfg (dict): Cấu hình khởi tạo cho fully connected layer
    """    
    def __init__(self, num_classes, in_channels, dropout=0.5, average_clips='prob', init_cfg=None):
        if init_cfg is None:
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)

        super(ClsHeadMalgo, self).__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.average_clips = average_clips

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Fully connected layer cho classification
        self.fc = nn.Linear(self.in_channels, self.num_classes)
        # Global average pooling trên cả 3 chiều (T, H, W)
        self.pool = nn.AdaptiveAvgPool3d(1)
        # Cross entropy loss cho classification
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass của classification head
        
        Args:
            x (torch.Tensor): Input features với shape [N, C, T, H, W]
                N: batch size
                C: channels
                T: temporal dimension
                H, W: spatial dimensions
        
        Returns:
            cls_scores (torch.Tensor): Classification scores với shape [N, num_classes]
        """        
        N, C, T, H, W = x.shape
        x = self.pool(x)
        x = x.view(N, C)
        assert x.shape[1] == self.in_channels

        if self.dropout is not None:
            x = self.dropout(x)

        cls_scores = self.fc(x)
        return cls_scores

    def loss(self, feats, data_samples):
        """
        Tính toán loss cho classification
        
        Args:
            feats (torch.Tensor): Input features
            data_samples (list): List of data samples chứa ground truth labels
            
        Returns:
            dict: Dictionary chứa classification loss
        """        
        cls_scores = self(feats)
        labels = torch.stack([x.gt_label for x in data_samples])
        labels = labels.squeeze()

        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)

        loss_cls = self.loss_fn(cls_scores, labels)
        return dict(loss_cls=loss_cls)

    def predict(self, feats, data_samples):
        """
        Thực hiện prediction và tính trung bình trên các clips
        
        Args:
            feats (torch.Tensor): Input features
            data_samples (list): List of data samples
            
        Returns:
            list: Data samples với prediction scores đã được cập nhật
        """        
        cls_scores = self(feats)
        num_views = cls_scores.shape[0] // len(data_samples)
        # assert num_views == data_samples[0].num_clips
        cls_scores = self.average_clip(cls_scores, num_views)

        for ds, sc in zip(data_samples, cls_scores):
            pred = LabelData(item=sc)
            ds.pred_scores = pred
        return data_samples

    def average_clip(self, cls_scores, num_views):
        """
        Tính trung bình scores từ nhiều clips của cùng một video
        
        Args:
            cls_scores (torch.Tensor): Classification scores từ mỗi clip
            num_views (int): Số lượng clips cho mỗi video
            
        Returns:
            torch.Tensor: Averaged classification scores
        """        
          
        if self.average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{self.average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        total_views = cls_scores.shape[0]
        cls_scores = cls_scores.view(total_views // num_views, num_views, -1)

        if self.average_clips is None:
            return cls_scores
        elif self.average_clips == 'prob':
            cls_scores = F.softmax(cls_scores, dim=2).mean(dim=1)
        elif self.average_clips == 'score':
            cls_scores = cls_scores.mean(dim=1)

        return cls_scores


@MODELS.register_module()
class RecognizerMalgo(BaseModel):
    """
    Mô hình nhận dạng video 3D hoàn chỉnh
    Kết hợp backbone network và classification head
    
    Args:
        backbone (dict): Cấu hình cho backbone network
        cls_head (dict): Cấu hình cho classification head
        data_preprocessor (dict): Cấu hình cho data preprocessing
    """    
    def __init__(self, backbone, cls_head, data_preprocessor):
        super().__init__(data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)
        self.cls_head = MODELS.build(cls_head)

    def extract_feat(self, inputs):
        """
        Trích xuất features từ input video
        
        Args:
            inputs (torch.Tensor): Input video tensor
            
        Returns:
            torch.Tensor: Extracted features
        """        
        inputs = inputs.view((-1, ) + inputs.shape[2:])
        return self.backbone(inputs)

    def loss(self, inputs, data_samples):
        """
        Tính toán loss cho mô hình
        
        Args:
            inputs (torch.Tensor): Input video tensor
            data_samples (list): List of data samples
            
        Returns:
            dict: Dictionary chứa loss values
        """        
        feats = self.extract_feat(inputs)
        loss = self.cls_head.loss(feats, data_samples)
        return loss

    def predict(self, inputs, data_samples):
        """
        Thực hiện prediction trên input video
        
        Args:
            inputs (torch.Tensor): Input video tensor
            data_samples (list): List of data samples
            
        Returns:
            list: Predictions cho mỗi sample
        """        
        feats = self.extract_feat(inputs)
        predictions = self.cls_head.predict(feats, data_samples)
        return predictions

    def forward(self, inputs, data_samples=None, mode='tensor'):
        """
        Forward pass của mô hình với các mode khác nhau
        
        Args:
            inputs (torch.Tensor): Input video tensor
            data_samples (list, optional): List of data samples
            mode (str): Mode hoạt động:
                - 'tensor': Trả về features
                - 'loss': Tính toán loss
                - 'predict': Thực hiện prediction
                
        Returns:
            Various: Kết quả tùy thuộc vào mode
        """        
        if mode == 'tensor':
            return self.extract_feat(inputs)
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode: {mode}')