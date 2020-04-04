from .registry import DATASETS
from .xml_style import XMLDataset
from detection.mmdet.core import eval_map, eval_recalls
# 注册安全帽数据集
from ..core import print_map_summary
import numpy as np
from  functools import reduce

@DATASETS.register_module
class SafeHatDataset(XMLDataset):
    # 分为安全帽和人类别
    CLASSES=('hat','person')

    def _readuce_func(self,x,y):
        z = []
        for index in range(len(self.CLASSES)):
            num_gts = x[index]['num_gts'] + y[index]['num_gts']
            num_dets = x[index]['num_dets'] + y[index]['num_dets']
            recall = x[index]['recall'] + y[index]['recall']
            precision = x[index]['precision'] +y[index]['precision']
            ap = x[index]['ap'] + y[index]['ap']
            z.append({
                'num_gts': num_gts,
                'num_dets': num_dets,
                'recall': recall,
                'precision': precision,
                'ap': ap
            })
        return  z

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                 scale_ranges=None):
        '''

        :param results:
        :param metric:
        :param logger:
        :param proposal_nums:
        :param iou_thr:
        :param scale_ranges:
        :return:
        '''
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        mean_aps = []
        other_infos = []
        if metric == 'mAP':
            for iou_thr_use in iou_thr:
                mean_ap, other_info = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr_use,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                other_infos.append(other_info)
        # 求所有ious_thr下的mAP
        mAP = np.array(mean_aps).mean().item()
        final_result = reduce(self._readuce_func,other_infos)
        for index in range(len(self.CLASSES)):
            final_result[index]['num_gts'] = final_result[index]['num_gts'] / len(iou_thr)
            final_result[index]['num_dets'] = final_result[index]['num_dets'] / len(iou_thr)
            final_result[index]['recall'] = final_result[index]['recall'] / len(iou_thr)
            final_result[index]['precision'] = final_result[index]['precision'] / len(iou_thr)
            final_result[index]['ap'] = final_result[index]['ap'] / len(iou_thr)
        print('coco mAP in iou_thrs:{0}'.format(iou_thr))
        print_map_summary(mAP,final_result,self.CLASSES,scale_ranges,logger)
        
        return {'mAP':mAP}

