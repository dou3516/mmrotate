# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List, Tuple

from mmengine.dataset import BaseDataset

from mmrotate.registry import DATASETS


@DATASETS.register_module()
class SHIPRS133Dataset(BaseDataset):
    """ShipRS 133 classes DOTA-like dataset for detection.

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
        ('Sovremenny-class destroyer', '052C-destroyer', 'Bunker', '636-hydrographic survey ship', '903A-replenishment ship', 'Tuzhong Class Salvage Tug', 'Traffic boat', '082II-Minesweeper', 'unknown', 'Emory S. Land-class submarine tender', 'Submarine', 'Barracks Ship', 'Whidbey Island-class dock landing ship', 'San Antonio-class amphibious transport dock', 'Arleigh Burke-class Destroyer', 'Ticonderoga-class cruiser', 'Barge', 'Sand Carrier', 'Oliver Hazard Perry-class frigate', 'Towing vessel', '022-missile boat', '037-submarine chaser', '904B-general stores issue ship', '072III-landing ship', '926-submarine support ship', 'Independence-class littoral combat ship', 'Avenger-class mine countermeasures ship', 'Mercy-class hospital ship', '052D-destroyer', '074-landing ship', '529-Minesweeper', 'USNS Bob Hope', '051-destroyer', 'Fishing Vessel', 'Freedom-class littoral combat ship', 'Nimitz Aircraft Carrier', 'Wasp-class amphibious assault ship', 'Sacramento-class fast combat support ship', 'Lewis and Clark-class dry cargo ship', '001-aircraft carrier', 'Xu Xiake barracks ship', 'Lewis B. Puller-class expeditionary mobile base ship', 'USNS Spearhead', '072A-landing ship', '081-Minesweeper', 'Takanami-class destroyer', '680-training ship', '920-hospital ship', '073-landing ship', 'Other Warship', '272-icebreaker', 'unknown auxiliary ship', '053H2G-frigate', '053H3-frigate', 'Container Ship', '053H1G-frigate', '903-replenishment ship', 'Yacht', 'Powhatan-class tugboat', 'YG-203 class yard gasoline oiler', 'YW-17 Class Yard Water', 'YO-25 class yard oiler', 'Asagiri-class Destroyer', 'Hiuchi-class auxiliary multi-purpose support ship', 'Henry J. Kaiser-class replenishment oiler', '072II-landing ship', '904-general stores issue ship', '056-corvette', '054A-frigate', '815-spy ship', '037II-missile boat', '037-hospital ship', '905-replenishment ship', '054-frigate', 'Abukuma-class destroyer escort', 'JMSDF LCU-2001 class utility landing crafts', 'Tenryu-class training support ship', 'Kurobe-class training support ship', 'Zumwalt-class destroyer', '071-amphibious transport dock', 'Tank ship', 'Iowa-class battle ship', 'Bulk carrier', 'Tarawa-class amphibious assault ship', '922A-Salvage lifeboat', 'Blue Ridge class command ship', '908-replenishment ship', '052B-destroyer', 'Hatsuyuki-class destroyer', 'Hatsushima-class minesweeper', 'Hyuga-class helicopter destroyer', 'Mashu-class replenishment oilers', 'Kongo-class destroyer', 'Towada-class replenishment oilers', 'Hatakaze-class destroyer', '891A-training ship', '721-transport boat', 'Akizuki-class destroyer', 'Osumi-class landing ship', 'Murasame-class destroyer', 'Uraga-class Minesweeper Tender', '909A-experimental ship', '074A-landing ship', '051C-destroyer', 'Hayabusa-class guided-missile patrol boats', '679-training ship', 'Forrestal-class Aircraft Carrier', 'Kitty Hawk class aircraft carrier', 'JS Suma', '909-experimental ship', 'Izumo-class helicopter destroyer', 'JS Chihaya', '639A-Hydroacoustic measuring ship', '815A-spy ship', 'North Transfer 990', 'Cyclone-class patrol ship', '052-destroyer', '917-lifeboat', '051B-destroyer', 'Yaeyama-class minesweeper', '635-hydrographic Survey Ship', 'USNS Montford Point', '925-Ocean salvage lifeboat', '648-submarine repair ship', '625C-Oceanographic Survey Ship', 'Sugashima-class minesweepers', 'Uwajima-class minesweepers', 'Northampton-class tug', 'Hibiki-class ocean surveillance ships', '055-destroyer', 'Futami-class hydro-graphic survey ships', 'JS Kurihama', '901-fast combat support ship'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128), (0, 192, 128), (128, 192, 128), (64, 64, 0), (192, 64, 0), (64, 192, 0), (192, 192, 0), (64, 64, 128), (192, 64, 128), (64, 192, 128), (192, 192, 128), (0, 0, 64), (128, 0, 64), (0, 128, 64), (128, 128, 64), (0, 0, 192), (128, 0, 192), (0, 128, 192), (128, 128, 192), (64, 0, 64), (192, 0, 64), (64, 128, 64), (192, 128, 64), (64, 0, 192), (192, 0, 192), (64, 128, 192), (192, 128, 192), (0, 64, 64), (128, 64, 64), (0, 192, 64), (128, 192, 64), (0, 64, 192), (128, 64, 192), (0, 192, 192), (128, 192, 192), (64, 64, 64), (192, 64, 64), (64, 192, 64), (192, 192, 64), (64, 64, 192), (192, 64, 192), (64, 192, 192), (192, 192, 192), (32, 0, 0), (160, 0, 0), (32, 128, 0), (160, 128, 0), (32, 0, 128), (160, 0, 128), (32, 128, 128), (160, 128, 128), (96, 0, 0), (224, 0, 0), (96, 128, 0), (224, 128, 0), (96, 0, 128), (224, 0, 128), (96, 128, 128), (224, 128, 128), (32, 64, 0), (160, 64, 0), (32, 192, 0), (160, 192, 0), (32, 64, 128), (160, 64, 128), (32, 192, 128), (160, 192, 128), (96, 64, 0), (224, 64, 0), (96, 192, 0), (224, 192, 0), (96, 64, 128), (224, 64, 128), (96, 192, 128), (224, 192, 128), (32, 0, 64), (160, 0, 64), (32, 128, 64), (160, 128, 64), (32, 0, 192), (160, 0, 192), (32, 128, 192), (160, 128, 192), (96, 0, 64), (224, 0, 64), (96, 128, 64), (224, 128, 64), (96, 0, 192), (224, 0, 192), (96, 128, 192), (224, 128, 192), (32, 64, 64), (160, 64, 64), (32, 192, 64), (160, 192, 64), (32, 64, 192), (160, 64, 192), (32, 192, 192), (160, 192, 192), (96, 64, 64), (224, 64, 64), (96, 192, 64), (224, 192, 64), (96, 64, 192), (224, 64, 192), (96, 192, 192), (224, 192, 192), (0, 32, 0), (128, 32, 0), (0, 160, 0), (128, 160, 0), (0, 32, 128)]
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
                osp.join(self.data_prefix['img_path'], '*.bmp'))
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
                img_name = img_id + '.bmp'
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
                            bbox_info = si.split(',')
                            instance['bbox'] = [float(i) for i in bbox_info[:8]]
                            cls_name = bbox_info[8]
                            instance['bbox_label'] = cls_map[cls_name]
                            difficulty = int(bbox_info[9])
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

