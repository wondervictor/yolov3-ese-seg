import logging
import warnings

import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


def flip_coef(bbox, size, flip_x=False, flip_y=False):

    if not len(size) == 2:
        raise ValueError("size requires length 2 tuple, given {}".format(len(size)))
    width, height = size
    bbox = bbox.copy()
    if flip_y:
        ymax = height - bbox[:, 1]
        ymin = height - bbox[:, 3]
        coef_center_y = height - bbox[:, 5]
        bbox[:, 1] = ymin
        bbox[:, 3] = ymax
        bbox[:, 5] = coef_center_y
    if flip_x:
        xmax = width - bbox[:, 0]
        xmin = width - bbox[:, 2]
        coef_center_x = width - bbox[:, 4]
        bbox[:, 0] = xmin
        bbox[:, 2] = xmax
        bbox[:, 4] = coef_center_x
    return bbox


def resize_coef(bbox, in_size, out_size):

    if not len(in_size) == 2:
        raise ValueError("in_size requires length 2 tuple, given {}".format(len(in_size)))
    if not len(out_size) == 2:
        raise ValueError("out_size requires length 2 tuple, given {}".format(len(out_size)))

    bbox = bbox.copy()
    x_scale = out_size[0] / in_size[0]
    y_scale = out_size[1] / in_size[1]

    bbox[:, 1] = y_scale * bbox[:, 1]
    bbox[:, 3] = y_scale * bbox[:, 3]
    bbox[:, 5] = y_scale * bbox[:, 5]

    bbox[:, 0] = x_scale * bbox[:, 0]
    bbox[:, 2] = x_scale * bbox[:, 2]
    bbox[:, 4] = x_scale * bbox[:, 4]

    return bbox


def val_resize_coef(bbox, in_size, out_size):

    if not len(in_size) == 2:
        raise ValueError("in_size requires length 2 tuple, given {}".format(len(in_size)))
    if not len(out_size) == 2:
        raise ValueError("out_size requires length 2 tuple, given {}".format(len(out_size)))

    bbox = bbox.copy()
    x_scale = out_size[0] / in_size[0]
    y_scale = out_size[1] / in_size[1]

    # bbox and polygon
    # bbox: [0,3] bounding box, [4, 364] polygon_x, [364, 724] polygon_y
    bbox[:, 1] = y_scale * bbox[:, 1]
    bbox[:, 3] = y_scale * bbox[:, 3]

    bbox[:, 0] = x_scale * bbox[:, 0]
    bbox[:, 2] = x_scale * bbox[:, 2]

    # resize polygon
    for i in range(360):
        bbox[:, 4 + 360 + i] = y_scale * bbox[:, 4 + 360 + i]
        bbox[:, 4 + i] = x_scale * bbox[:, 4 + i]

    return bbox


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


class SBDVOCTrainDataset(Dataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, root='', image_size=416, index_map=None):
        self._root = root
        self._image_size = image_size
        self._im_shapes = {}
        self._anno_path = os.path.join('{}', './cheby_fit/n8_xml', '{}.xml')
        self._image_path = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self.classes = self.CLASSES
        self.num_class = len(self.CLASSES)
        self.split = [('sbdche', 'train_8_bboxwh')]
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._items = self._load_items(self.split)

    def __len__(self):
        return len(self._items)

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for year, name in splits:
            root = os.path.join(self._root, 'VOC' + str(year))
            lf = os.path.join(root, 'ImageSets', 'Segmentation', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(root, line.strip()) for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            obj_label_info = []
            difficult = int(obj.find('difficult').text)
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text))
            ymin = (float(xml_box.find('ymin').text))
            xmax = (float(xml_box.find('xmax').text))
            ymax = (float(xml_box.find('ymax').text))
            coef_center_x = (float(xml_box.find('coef_center_x').text))
            coef_center_y = (float(xml_box.find('coef_center_y').text))
            xml_coef = obj.find('coef').text
            xml_coef = xml_coef.split()
            coef = [float(xml_coef[i]) for i in range(len(xml_coef))]
            obj_label_info.append(xmin)
            obj_label_info.append(ymin)
            obj_label_info.append(xmax)
            obj_label_info.append(ymax)
            obj_label_info.append(coef_center_x)
            obj_label_info.append(coef_center_y)
            for i in range(len(coef)):
                obj_label_info.append(coef[i])
            obj_label_info.append(cls_id)
            obj_label_info.append(difficult)
            obj_label_info.append(width)
            obj_label_info.append(height)
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append(obj_label_info)
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _validate_class_names(self, class_list):
        """Validate class names."""
        assert all(c.islower() for c in class_list), "uppercase characters"
        stripped = [c for c in class_list if c.strip() != c]
        if stripped:
            warnings.warn('white space removed for {}'.format(stripped))

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img = transforms.Resize(size=(self._image_size, self._image_size))(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(transforms.ToTensor()(img))
        input_size = (w, h)
        output_size = (self._image_size, self._image_size)

        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        label = self._load_label(idx)

        label = resize_coef(label, input_size, output_size)

        boxes = label[:, :4]
        target = np.zeros((len(label), 26))
        target[:, 1] = label[:, -4]
        target[:, 2] = (boxes[:, 0] + boxes[:, 2])/2
        target[:, 3] = (boxes[:, 1] + boxes[:, 3])/2
        target[:, 4] = boxes[:, 2] - boxes[:, 0]
        target[:, 5] = boxes[:, 3] - boxes[:, 1]
        target[:, 2:6] = target[:, 2:6] / self._image_size

        coef_centers = label[:, 4:6] / self._image_size
        target[:, 6:8] = coef_centers
        target[:, 8:] = label[:, 6:24]

        if random.random() < 0.5:
            img = torch.flip(img, [-1])
            target[:, 2] = 1 - target[:, 2]
            target[:, 6] = 1 - target[:, 6]

        target = torch.from_numpy(target)
        return img_path, img, target

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0).float()
        # Selects new image size every tenth batch
        # if self.multiscale and self.batch_count % 10 == 0:
        #    self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        # imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        # self.batch_count += 1
        imgs = torch.stack(imgs)
        return paths, imgs, targets


class SBDVOCValDataset(Dataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, root='', image_size=416, index_map=None):
        self._root = root
        self._image_size = image_size
        self._im_shapes = {}
        self._anno_path = os.path.join('{}', './label_polygon_360_xml', '{}.xml')
        self._image_path = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self.classes = self.CLASSES
        self.num_class = len(self.CLASSES)
        self.split = [('sbdche', 'val_8_bboxwh')]
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._items = self._load_items(self.split)

    def __len__(self):
        return len(self._items)

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for year, name in splits:
            root = os.path.join(self._root, 'VOC' + str(year))
            lf = os.path.join(root, 'ImageSets', 'Segmentation', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(root, line.strip()) for line in f.readlines()]
        return ids

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _validate_class_names(self, class_list):
        """Validate class names."""
        assert all(c.islower() for c in class_list), "uppercase characters"
        stripped = [c for c in class_list if c.strip() != c]
        if stripped:
            warnings.warn('white space removed for {}'.format(stripped))

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            obj_label_info = []
            difficult = int(obj.find('difficult').text)
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text))
            ymin = (float(xml_box.find('ymin').text))
            xmax = (float(xml_box.find('xmax').text))
            ymax = (float(xml_box.find('ymax').text))
            xml_points_x = obj.find('points_x').text
            xml_points_x = xml_points_x.split()
            points_x = [float(xml_points_x[i]) for i in range(len(xml_points_x))]
            xml_points_y = obj.find('points_y').text
            xml_points_y = xml_points_y.split()
            points_y = [float(xml_points_y[i]) for i in range(len(xml_points_y))]
            obj_label_info.append(xmin)
            obj_label_info.append(ymin)
            obj_label_info.append(xmax)
            obj_label_info.append(ymax)
            for i in range(len(points_x)):
                obj_label_info.append(points_x[i])
            for i in range(len(points_y)):
                obj_label_info.append(points_y[i])
            obj_label_info.append(cls_id)
            obj_label_info.append(difficult)
            obj_label_info.append(width)
            obj_label_info.append(height)
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append(obj_label_info)
        return np.array(label)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img = transforms.Resize(size=(self._image_size, self._image_size))(img)
        img = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])(transforms.ToTensor()(img))
        input_size = (w, h)
        output_size = (self._image_size, self._image_size)

        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        label = self._load_label(idx)

        label = val_resize_coef(label, input_size, output_size)
        target = np.zeros((len(label), 726))
        target[:, 1] = label[:,  -4]
        target[:, 2:6] = label[:, :4]
        target[:, 6:726] = label[:, 4:724]
        target = torch.from_numpy(target)
        return img_path, img, target

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0).float()
        # Selects new image size every tenth batch
        # if self.multiscale and self.batch_count % 10 == 0:
        #    self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        # imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        # self.batch_count += 1
        imgs = torch.stack(imgs)
        # print(targets[..., 2:6])
        return paths, imgs, targets
