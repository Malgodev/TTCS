import os.path as osp
from mmengine.fileio import list_from_file
from mmengine.dataset import BaseDataset
from mmaction.registry import DATASETS

@DATASETS.register_module()
class DatasetMalgo(BaseDataset):
    """
    Dataset: Tập dữ liệu được dùng để train, validate mô hình.
    """
    def __init__(self, ann_file, pipeline, data_root, data_prefix=dict(video=''),
                 test_mode=False, modality='RGB', **kwargs):
        """
        Nhận đường dẫn file và và các thông tin cần thiết.
        Args:
            ann_file: đường dẫn tới annotation file
            pipeline: pipeline
            data_root: đường dẫn tới thư mục chứa dữ liệu
            data_prefix: đường dẫn tới thư mục chứa video
            test_mode: True nếu là test mode
            modality: RGB hoặc Flow
            **kwargs: các thông số khác
        """
        # modality: loại dữ liệu mà hệ thống sử dụng nhằm nhận dạng và phân loại
        # có thể là RGB hoặc Flow
        self.modality = modality 
        super(DatasetMalgo, self).__init__(ann_file=ann_file, pipeline=pipeline, data_root=data_root,
                                           data_prefix=data_prefix, test_mode=test_mode,
                                           **kwargs)

    def load_data_list(self):
        """
        Đọc dữ liệu từ file annotation
        """
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split()
            filename, label = line_split
            label = int(label)
            filename = osp.join(self.data_prefix['video'], filename)
            data_list.append(dict(filename=filename, label=label))
        return data_list

    def get_data_info(self, idx: int) -> dict:
        """
        Lấy dữ liệu dựa theo index
        """
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        return data_info