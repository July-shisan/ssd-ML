"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

# VOC_CLASSES = (  # always index 0
#     'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor')
VOC_CLASSES = (
    '带电芯充电宝', '不带电芯充电宝'
)

# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join(HOME, "VOCdevkit/")


# class VOCAnnotationTransform(object):
#     """Transforms a VOC annotation into a Tensor of bbox coords and label index
#     Initilized with a dictionary lookup of classnames to indexes
#
#     Arguments:
#         class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
#             (default: alphabetic indexing of VOC's 20 classes)
#         keep_difficult (bool, optional): keep difficult instances or not
#             (default: False)
#         height (int): height
#         width (int): width
#     """
#
#     def __init__(self, class_to_ind=None, keep_difficult=False):
#         self.class_to_ind = class_to_ind or dict(
#             zip(VOC_CLASSES, range(len(VOC_CLASSES))))
#         self.keep_difficult = keep_difficult
#
#     def __call__(self, target, width, height):
#         """
#         Arguments:
#             target (annotation) : the target annotation to be made usable
#                 will be an ET.Element
#         Returns:
#             a list containing lists of bounding boxes  [bbox coords, class name]
#         """
#         res = []
#         for obj in target.iter('object'):
#             difficult = int(obj.find('difficult').text) == 1
#             if not self.keep_difficult and difficult:
#                 continue
#             name = obj.find('name').text.lower().strip()
#             bbox = obj.find('bndbox')
#
#             pts = ['xmin', 'ymin', 'xmax', 'ymax']
#             bndbox = []
#             for i, pt in enumerate(pts):
#                 cur_pt = int(bbox.find(pt).text) - 1
#                 # scale height or width
#                 cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
#                 bndbox.append(cur_pt)
#             label_idx = self.class_to_ind[name]
#             bndbox.append(label_idx)
#             res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
#             # img_id = target.find('filename').text[:-4]
#
#         return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]
class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult
        # 添加的记录所有小类总数
        self.type_dict = {}
        # 记录大类数量
        self.type_sum_dict = {}

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
            it has been changed to the path of annotation-2019-07-10
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        # 遍历Annotation
        # root_annotation = '/media/dsg3/datasets/Xray20190704/Annotation/'
        res = []
        with open(target, "r", encoding='utf-8') as f1:
            dataread = f1.readlines()
        for annotation in dataread:
            bndbox = []
            temp = annotation.split()
            name = temp[1]
            # 只读两类
            if name != '带电芯充电宝' and name != '不带电芯充电宝':
                continue
            xmin = int(temp[2]) / width
            # 只读取V视角的
            if xmin > 1:
                continue
            if xmin < 0:
                xmin = 0
            ymin = int(temp[3]) / height
            if ymin < 0:
                ymin = 0
            xmax = int(temp[4]) / width
            if xmax > 1:  # 是这么个意思吧？
                xmax = 1
            ymax = int(temp[5]) / height
            if ymax > 1:
                ymax = 1
            bndbox.append(xmin)
            bndbox.append(ymin)
            bndbox.append(xmax)
            bndbox.append(ymax)
            label_idx = self.class_to_ind[name]
            # label_idx = name
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        if len(res) == 0:
            return [[0, 0, 0, 0, 3]]
        return res

class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 # image_sets=[('2007', 'trainval')],
                 image_sets='trainval',
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        # self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        # self._annopath = osp.join('%s', 'Annotations', '%s.txt')
        self._annopath = osp.join('%s' % self.root, 'Annotations', '%s.txt')
        self._imgpath = osp.join('%s' % self.root, 'JPEGImages', '%s.TIFF')
        ###这尼玛还有小写的tiff？
        self._imgpath1 = osp.join('%s' % self.root, 'JPEGImages', '%s.tiff')
        self._imgpath_jpg = osp.join('%s' % self.root, 'JPEGImages', '%s.jpg')
        # self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()

        with open(osp.join(self.root, 'ImageSets', 'Main', image_sets + '.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.ids.append(line.strip('\n'))
        # for (year, name) in image_sets:
        #     rootpath = osp.join(self.root, 'VOC' + year)
        #     for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #         self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w, og_im = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        # target = ET.parse(self._annopath % img_id).getroot()
        target = self._annopath % img_id
        img = cv2.imread(self._imgpath % img_id)
        if img is None:
            img = cv2.imread(self._imgpath1 % img_id)
        if img is None:
            img = cv2.imread(self._imgpath_jpg % img_id)
        if img is None:
            print('\nwrong\n')

        height, width, channels = img.shape
        og_img = img

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, og_img
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id)
        if img is None:
            img = cv2.imread(self._imgpath1 % img_id)
        if img is None:
            img = cv2.imread(self._imgpath_jpg % img_id)
        if img is None:
            print('\nwrong\n')

        height, width, channels = img.shape
        target = self._annopath % img_id

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
        # return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        return img

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        # anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._annopath % img_id
        gt = self.target_transform(anno, 1, 1)

        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
