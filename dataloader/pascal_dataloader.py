import os
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# self-define class
import custom_transforms
# debug
import matplotlib.pyplot as plt

category_names = ['background',
                 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                 'bus', 'car', 'cat', 'chair', 'cow',
                 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class VOC_Instance_Segmentation(Dataset):
    '''
    VOC instance segmentation
    * mask data as below
        \- 0:background
        \- 255: object's contour
        \- 1~n: objects ID
    TODO: pick object size, and download
    '''

    def __init__(self,
                 root='/home/riq/segmentation/benchmark/VOC/VOCdevkit/VOC2012',
                 split_sets='train',
                 transform=None,
                 transform_handcraft=None,
                 #download=False,
                 preprocess=False,
                 area_thres=0
                 ):

        self.root = root
        self.transform = transform
        self.area_thres = area_thres
        self.transform_handcraft = transform_handcraft

        # split set
        if isinstance(split_sets, list):
            split_sets = ''.join(split_sets)

        # get file path of image and ground truth in specific set i.e. train.txt, trainval.txt, val.txt
        self.images, self.instance_objs = self._get_pair_path(split_sets)

        # get file path of image and objects(from ground truth)
        self.pair_list = self._get_instance_obj()
        print('INFO: number of data', len(self.pair_list))

    def __getitem__(self, index):
        to_PIL = transforms.ToPILImage()
        ID, img_path, gt_path = self.pair_list[index]
        img, gt = Image.open(img_path), Image.open(gt_path)
        gt = to_PIL((np.array(gt)==ID).astype(np.float32))


        if self.transform:
            img, gt = self.transform([img, gt])
            input_pair = {'image':img, 'gt': gt}

        if self.transform_handcraft:
            img, gt = self.transform_handcraft.ObjectCenterCrop(gt, img)
            img, gt = self.transform_handcraft.Fix_size(img, gt)
            # input_pair = {'image':img, 'gt': gt, }

            heat_map = self.transform_handcraft.get_extreme_point_channel(gt)
            concate_input = self.transform_handcraft.concat_inputs(img, heat_map)
            # concate_input = self.transform_handcraft.ToTensor(concate_input)
            # gt = self.transform_handcraft.ToTensor(gt)
            input_pair = {'image':img, 'gt':gt, 'heat_map':heat_map, 'input':concate_input}

        return input_pair
    
    def __len__(self):
        return len(self.pair_list)

    def _get_instance_obj(self):
        # TODO: threshold for object size 
        area_th_str = ""
        if self.area_thres != 0:
            area_th_str = '_area_thres-' + str(self.area_thres)
        
        pair_list = []
        for img_path, gt_path in zip(self.images, self.instance_objs):
            gt = Image.open(gt_path)
            object_ID = np.unique(gt)[1:-1]
            for ID in object_ID:
                pair_list.append([ID, img_path, gt_path])
        return pair_list

    def _get_pair_path(self, split_sets):
        images_path, instance_objs_path = [], []

        # A File denote image belong to which set        
        split_set_dir = os.path.join(self.root, 'ImageSets', 'Segmentation')
        seg_obj_dir = os.path.join(self.root, 'SegmentationObject')
        image_dir = os.path.join(self.root, 'JPEGImages')        

        # Read img name from whole set of .txt file
        # Can't use glob, because we don't want to load all image in the folder at the same time
        with open(os.path.join(os.path.join(split_set_dir, split_sets + '.txt')), "r") as f:
            img_names = f.read().splitlines()

        for img_name in img_names:
            image = os.path.join(image_dir, img_name + ".jpg")
            seg_obj = os.path.join(seg_obj_dir, img_name + ".png")

            assert os.path.isfile(image)
            assert os.path.isfile(seg_obj)

            images_path.append(image)
            instance_objs_path.append(seg_obj)
        
        assert (len(images_path) == len(instance_objs_path))
        return images_path, instance_objs_path   

def torch_VOC():
    '''
    VOC SegmentationClass include (background, multi object, contour)
    Mask which load by PIL Image value is as below
    0: background
    1~20: object ID
    255: contour
    Pytorch only supports semantic segmentation output pair so far, and didn't split contour
    '''
    voc = torchvision.datasets.VOCSegmentation('./data', year='2012', image_set='trainval', download=False)
    print(type(voc[0]))
    # # loop all pair
    # for sample in voc:
    #     img, gt = sample[0], sample[1]
    #     if np.unique(np.array(gt))[-1] != 255:
    #         print("INFO: WTF")

    # single pair
    sample = voc[0]
    img, gt = sample[0], sample[1]

    # gt = np.array(gt)
    print('INFO: Ground truth object ID {}'.format(np.unique(gt)))
    print('INFO: 0=background, 255=contour, other=object ID')

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(np.array(gt) * 255) # show object ID
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # delete()
    # torch_VOC()

    # 1.Show transforms's result before transfer to tensor
    # execute by PIL image
    data_transforms = transforms.Compose([
        custom_transforms.RandomHorizontalFlip(1.0),
        custom_transforms.RandomRotation((-20,20), scales=(0.75, 1.0)),
        custom_transforms.to_numpy()
    ])

    # excute by numpy image
    handcraft_transforms = custom_transforms.Compose_dict([
        custom_transforms.ObjectCenterCrop(),
        custom_transforms.Fix_size(size=512),
        custom_transforms.get_extreme_point_channel(),
        custom_transforms.concat_inputs()
    ])

    voc = VOC_Instance_Segmentation(split_sets=['train'], transform=data_transforms, \
                                    transform_handcraft=handcraft_transforms)
    for ii, dct in enumerate(voc):
        img, gt, concate = dct['image'], dct['gt'], dct['heat_map']
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.subplot(1, 3, 2)
        plt.imshow(gt)
        plt.subplot(1, 3, 3)
        plt.imshow(concate)
        plt.tight_layout()
        plt.pause(5)

        if ii > 10:
            exit()

    # 2.Show transforms's result tensor size
    handcraft_transforms = custom_transforms.Compose_dict([
        custom_transforms.ObjectCenterCrop(),
        custom_transforms.Fix_size(size=512),
        custom_transforms.get_extreme_point_channel(),
        custom_transforms.concat_inputs(),
        transforms.ToTensor()
    ])

    voc = VOC_Instance_Segmentation(split_sets=['train'], transform=data_transforms, \
                                    transform_handcraft=handcraft_transforms)
    for ii, dct in enumerate(voc):
        img, gt, concate = dct['image'], dct['gt'], dct['input']
        print('shape of tensor {}'.format(concate.shape))

        if ii > 10:
            exit()