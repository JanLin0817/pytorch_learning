import sys
sys.path.append('/home/riq/segmentation/benchmark')
sys.path.append('/home/riq/segmentation/benchmark/cocoapi/PythonAPI')

import os
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from torch.utils.data import Dataset
# from mypath import Path #TODO
from PIL import Image, ImageFile

# from abc import ABCMeta, abstractmethod # metaclass=ABCMeta

class COCOSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 #base_dir=Path.db_root_dir('coco'),
                 base_dir='',
                 split='train',
                 year='2014',
                 transform=None,
                ):
        """
        :param base_dir: path to COCO dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        print('FATHER')
        base_dir = '/home/iDL_public/DATASET/Public_Dataset/ADAS/BoundingBoxOrTracking/coco'
        base_dir = os.path.join(base_dir, 'coco{}'.format(year))
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, '{}{}'.format(split, year))
        self._annot_dir = os.path.join(self._base_dir, 'annotations', 'instances_{}{}.json'.format(split, year))
        self.year = year
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        self.transform = transform

        self.img_ids = []
        self.objects = []

        # coco api load annotation
        coco = COCO(self._annot_dir)
        self.annToMask = coco.annToMask
        class_ids = sorted(coco.getCatIds())
        
        # get image id for all kind of categories
        for i in class_ids:
            self.img_ids.extend(list(coco.getImgIds(catIds=[i])))

        # Remove duplicates
        self.img_ids = list(set(self.img_ids))

        for img_id in self.img_ids:
            width = coco.imgs[img_id]["width"]
            height = coco.imgs[img_id]["height"]
            img_path = os.path.join(self._image_dir, coco.imgs[img_id]['file_name'])
            ann_infos = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=None))

            for ann_info in ann_infos:
                objects = {
                    "image_path": img_path, \
                    "mask_annotation": ann_info, \
                    "height": height, \
                    "width": width, \
                }
                self.objects.append(objects)

        # Display stats
        print('Number of images: {:d}\nNumber of objects: {:d}'.format(len(self.img_ids), len(self.objects)))

    def __len__(self):
        return len(self.objects)

    #　@abstractmethod
    def self_define_transform(self, sample):
        return sample

    def _make_img_gt_point_pair(self, index):
        object = self.objects[index]
        image_path = object['image_path']
        annotation = object['mask_annotation']
        height = object['height']
        width = object['width']

        _target = self.annToMask(annotation)
        
        if annotation['iscrowd']:
            # For crowd masks, annToMask() sometimes returns a mask
            # smaller than the given dimensions. If so, resize it.
            if _target.shape[0] != height or _target.shape[1] != width:
                _target = np.ones([height, width], dtype=bool)

        # Read Image
        _img = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)

        return _img, _target

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target}

        if self.transform is not None:
            sample = self.self_define_transform(sample)

        return sample

    def __str__(self):
        return 'COCO' + str(self.year) + '(split=' + str(self.split)

class coco_instance_seg(COCOSegmentation):
    def __init__(self,
                 #base_dir=Path.db_root_dir('coco'),
                 base_dir='',
                 split='train',
                 year='2014',
                 transform=None,
                ):
        """
        :param base_dir: path to COCO dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super(coco_instance_seg, self).__init__(base_dir,
                 split,
                 year,
                 transform=transform,)

    def self_define_transform(self, sample):
        return sample

if __name__ == '__main__':

    # test coco segmentation
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    voc_train = coco_instance_seg(year='2017', split='val', transform='a')
    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=4)

    for imput_pair in voc_train:
        img, gt = imput_pair['image'], imput_pair['gt']
        plt.subplot(1,2,1)
        plt.imshow(img/255)
        plt.subplot(1,2,2)
        plt.imshow(gt)
        plt.pause(0.5)

    exit()