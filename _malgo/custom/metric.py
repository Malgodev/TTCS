import copy
from collections import OrderedDict
from mmengine.evaluator import BaseMetric
from mmaction.evaluation import top_k_accuracy
from mmaction.registry import METRICS


@METRICS.register_module()
class AccuracyMetric(BaseMetric):
    def __init__(self, topk=(1, 5), collect_device='cpu', prefix='acc'):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.topk = topk

    def process(self, data_batch, data_samples):
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            scores = data_sample['pred_scores']['item'].cpu().numpy()
            label = data_sample['gt_label'].item()
            result['scores'] = scores
            result['label'] = label
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        eval_results = OrderedDict()
        labels = [res['label'] for res in results]
        scores = [res['scores'] for res in results]
        topk_acc = top_k_accuracy(scores, labels, self.topk)
        for k, acc in zip(self.topk, topk_acc):
            eval_results[f'topk{k}'] = acc
        return eval_results