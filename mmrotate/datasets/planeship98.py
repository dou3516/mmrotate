# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List, Tuple

from mmengine.dataset import BaseDataset

from mmrotate.registry import DATASETS


@DATASETS.register_module()
class PLANESHIP98Dataset(BaseDataset):
    """PLANESHIP98 classes DOTA-like dataset for detection.

    Note: ``ann_file`` in DOTADataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTADataset,
    it is the path of a folder containing XML files.

    Args:
        img_shape (tuple[int]): The shape of images. Due to the huge size
            of the remote sensing image, we will cut it into slices with
            the same shape. Defaults to (1024, 1024).
        diff_thr (int): The difficulty threshold of ground truth. Bboxes
            with difficulty higher than it will be ignored. The range of this
            value should be non-negative integer. Defaults to 100.
    """

    METAINFO = {
        'classes':
        ('P7', 'F5', 'C12', 'F6', 'C1', 'K5', 'S7', 'P3', 'C10', 'B1', 'S4', 'C2', 'W1', 'N1', 'T5', 'M5', 'P6', 'E2', 'A7', 'S5', 'S6', 'D2', 'C4', 'T2', 'E3', 'S1', 'C7', 'F1', 'A4', 'C13', 'V2', 'S3', 'V1', 'F2', 'C9', 'H1', 'W2', 'F8', 'R3', 'E1', 'P5', 'T9', 'U1', 'H3', 'T7', 'T3', 'K1', 'R1', 'L1', 'C15', 'A9', 'A6', 'C6', 'C5', 'B3', 'M1', 'S2', 'M3', 'T1', 'P1', 'B2', 'K4', 'A8', 'T8', 'M4', 'Y1', 'A3', 'A1', 'K3', 'T10', 'C3', 'A2', 'L3', 'W3', 'P2', 'U2', 'P4', 'F3', 'M2', 'T4', 'C8', 'R4', 'H2', 'D1', 'T6', 'I1', 'L4', 'A5', 'E4', 'R2', 'C16', 'C14', 'Z1', 'C11', 'T11', 'K2', 'F7', 'L2'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128), (0, 192, 128), (128, 192, 128), (64, 64, 0), (192, 64, 0), (64, 192, 0), (192, 192, 0), (64, 64, 128), (192, 64, 128), (64, 192, 128), (192, 192, 128), (0, 0, 64), (128, 0, 64), (0, 128, 64), (128, 128, 64), (0, 0, 192), (128, 0, 192), (0, 128, 192), (128, 128, 192), (64, 0, 64), (192, 0, 64), (64, 128, 64), (192, 128, 64), (64, 0, 192), (192, 0, 192), (64, 128, 192), (192, 128, 192), (0, 64, 64), (128, 64, 64), (0, 192, 64), (128, 192, 64), (0, 64, 192), (128, 64, 192), (0, 192, 192), (128, 192, 192), (64, 64, 64), (192, 64, 64), (64, 192, 64), (192, 192, 64), (64, 64, 192), (192, 64, 192), (64, 192, 192), (192, 192, 192), (32, 0, 0), (160, 0, 0), (32, 128, 0), (160, 128, 0), (32, 0, 128), (160, 0, 128), (32, 128, 128), (160, 128, 128), (96, 0, 0), (224, 0, 0), (96, 128, 0), (224, 128, 0), (96, 0, 128), (224, 0, 128), (96, 128, 128), (224, 128, 128), (32, 64, 0), (160, 64, 0), (32, 192, 0), (160, 192, 0), (32, 64, 128), (160, 64, 128), (32, 192, 128), (160, 192, 128), (96, 64, 0), (224, 64, 0), (96, 192, 0), (224, 192, 0), (96, 64, 128), (224, 64, 128), (96, 192, 128), (224, 192, 128), (32, 0, 64), (160, 0, 64)]
    }

    def __init__(self,
                 img_shape: Tuple[int, int] = (1024, 1024),
                 diff_thr: int = 100,
                 **kwargs) -> None:
        self.img_shape = img_shape
        self.diff_thr = diff_thr
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], '*.tif'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id
                data_info['height'] = self.img_shape[0]
                data_info['width'] = self.img_shape[1]

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in '
                                 f'{self.ann_file}')
            for txt_file in txt_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + '.tif'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)
                data_info['height'] = self.img_shape[0]
                data_info['width'] = self.img_shape[1]

                instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        if not (si.startswith('ima') or si.startswith('gsd')):
                            instance = {}
                            bbox_info = si.split(' ')
                            instance['bbox'] = [float(i) for i in bbox_info[:8]]
                            cls_name = bbox_info[8].strip('\n')
                            instance['bbox_label'] = cls_map[cls_name]
                            difficulty = 0  # int(bbox_info[9])
                            if difficulty > self.diff_thr:
                                instance['ignore_flag'] = 1
                            else:
                                instance['ignore_flag'] = 0
                            instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get DOTA category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]

