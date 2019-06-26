
import cv2
import random
import numpy as np
import torchvision.transforms.functional as TF

def extreme_points(mask, relax):
    '''
    random point around top, bottom, left, right within pixel difference < relax
    '''
    def random_point(x_list, y_list, idx_list):
        idx_list = idx_list[0]
        idx = random.randint(0, len(idx_list) - 1)
        idx = idx_list[idx]
        return [x_list[idx], y_list[idx]]

    # List of points of object mask 
    pt_y, pt_x = np.where(mask > 0.5) 

    # Find extreme points
    return np.array([random_point(pt_x, pt_y, np.where(pt_x <= np.min(pt_x) + relax)), # left
                     random_point(pt_x, pt_y, np.where(pt_x >= np.max(pt_x) - relax)), # right
                     random_point(pt_x, pt_y, np.where(pt_y <= np.min(pt_y) + relax)), # top
                     random_point(pt_x, pt_y, np.where(pt_y >= np.max(pt_y) - relax))  # bottom
                     ])

def gaussian_filter(size=(100,100), center=(20,20), sigma=10):
    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    x = x[np.newaxis, :]
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-0.5 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(np.float64)

def get_bbox(mask, crop_range=0):
    mask_index = np.where(mask > 0)

    if mask_index[0].shape[0] == 0:
        return None

    y_min, x_min = 0, 0
    x_max = mask.shape[1] - 1
    y_max = mask.shape[0] - 1

    x_min = max(mask_index[1].min() - crop_range, x_min)
    y_min = max(mask_index[0].min() - crop_range, y_min)
    x_max = min(mask_index[1].max() + crop_range, x_max)
    y_max = min(mask_index[0].max() + crop_range, y_max)

    # return x_min, y_min, x_max, y_max
    return y_min, y_max, x_min, x_max

class RandomHorizontalFlip:
    '''
    random flip img horizontally with probability as 'flip_prob'
    '''
    def __init__(self, flip_prob=0.5):
        self.prob = flip_prob

    def __call__(self, args):
        output = []
        if random.random() < self.prob:
            for data in args:
                output.append(TF.hflip(data))
        return output

class RandomRotation:
    def __init__(self, angles=(-20, 20), scales=(0.75, 1.25)):
        self.angles = angles
        self.scales = scales
    
    def __call__(self, args):
        output = []
        random_angle = (self.angles[1] - self.angles[0]) * random.random() - \
                       (self.angles[1] - self.angles[0])/2
        random_scale = (self.scales[1] - self.scales[0]) * random.random() - \
                       (self.scales[1] - self.scales[0]) / 2 + 1                       
        for data in args:
            output.append(TF.affine(data, angle=random_angle, translate=(0,0), scale=random_scale, shear=0))
            
        return output

class to_numpy:
    '''
    Transfer PIL image to numpy 
    gray image as shape = h
    '''
    def __init__(self):
        pass
    def __call__(self, args):
        output = []
        for data in args:
            data = np.array(data)
            if len(data.shape) > 2:
                data = np.squeeze(data)
            output.append(data)
        return output

class ObjectCenterCrop:
    '''
    use object in mask as center and crop the object
    crop_range: left, right, top, bottom most pixel relax crop_range \
                i.e.: left_most = 52, crop_range = 50, new_left_most = 2
    pad: pad zero if *_most to boundary length less than crop_range
    TODO: 座標她媽對不到
    '''
    def __init__(self, crop_range=50, pad=True):
        self.crop_range = crop_range
        self.pad = pad

    def __call__(self, mask, img):

        box = get_bbox(mask, 0)
        if box == None:
            return img, mask

        if self.pad == False:
            box = get_bbox(mask, self.crop_range)
            mask = mask[box[0]:box[1], box[2]:box[3]]
            img = img[box[0]:box[1], box[2]:box[3]]
            return img, mask

        y_min, y_max, x_min, x_max = box
        box = np.array([y_min, y_max, x_min, x_max])
        pad = np.absolute(box - np.array([0, mask.shape[0], 0, mask.shape[1]]))
        pad = self.crop_range - np.minimum(pad, self.crop_range)

        pad = pad.reshape(2,2)
        mask = np.pad(mask, pad, 'constant', constant_values=((0,0),(0,0)))

        pad_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for ii in range(img.shape[-1]):
            pad_img[:,:,ii] = np.pad(img[:,:,ii], pad, 'constant', constant_values=0)

        box_0 = get_bbox(mask, crop_range=50)
        mask = mask[box_0[0]:box_0[1], box_0[2]:box_0[3]]
        pad_img = pad_img[box_0[0]:box_0[1], box_0[2]:box_0[3]]

        return pad_img, mask

class Fix_size:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img, gt):
        gt = cv2.resize(gt, self.size, interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_CUBIC)

        return img, gt

class get_extreme_point_channel(object):
    """
    Create a Gaussian images with value from 0 ~ 255
    Gaussian center are points
    """
    def __init__(self, relax=0, sigma=10, self_max=255.):
        self.relax = relax
        self.sigma = sigma
        self.max = self_max

    def __call__(self, mask):
        # transfer to numpy array
        # mask = np.array(mask)
        
        # if there are no objects in mask skip
        if np.max(mask) == 0:
            return np.zeros(mask.shape[:2])

        # get for points on mask contour
        points = extreme_points(mask, self.relax)

        # pu gaussian on points
        mask = np.zeros(mask.shape[:2])
        for pt in points:
            mask = np.maximum(mask, gaussian_filter(size=mask.shape[:2], center=pt, sigma=self.sigma))

        # normalize to 0~255
        mask = self.max * (mask - mask.min()) / (mask.max() - mask.min() + 1e-10)
        return mask

class concat_inputs(object):
    '''
    img shape as (H,W,C)
    concate img with heat_map with axis=C 
    '''
    def __init__(self):
        pass

    def __call__(self, img, args):
        if not isinstance(args, list):
            args = [args]

        for data in args:
            if len(data.shape) == 2:
                data = data[:,:, np.newaxis]

            assert len(data.shape) == 3

            img = np.concatenate((np.array(img), data), axis=2)

        return img

class Compose_dict:

    def __init__(self, transforms):
        func_dct = {}
        for func in transforms:
            func_dct[func.__class__.__name__] = func
        self.__dict__.update(func_dct)