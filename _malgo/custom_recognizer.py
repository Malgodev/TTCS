import torch
from mmengine.model import BaseDataPreprocessor, stack_batch
from mmaction.registry import MODELS


@MODELS.register_module()
class DataPreprocessorZelda(BaseDataPreprocessor):
    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer(
            'mean',
            torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1, 1),
            False)
        self.register_buffer(
            'std',
            torch.tensor(std, dtype=torch.float32).view(-1, 1, 1, 1),
            False)

    def forward(self, data, training=False):
        data = self.cast_data(data)
        inputs = data['inputs']
        batch_inputs = stack_batch(inputs)  # Batching
        batch_inputs = (batch_inputs - self.mean) / self.std  # Normalization
        data['inputs'] = batch_inputs
        return data
    
from mmaction.registry import MODELS

data_preprocessor_cfg = dict(
    type='DataPreprocessorZelda',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375])

data_preprocessor = MODELS.build(data_preprocessor_cfg)

preprocessed_inputs = data_preprocessor(batched_packed_results)
print(preprocessed_inputs['inputs'].shape)