from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.voc_polygon_detection import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from IPython import embed


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = SBDVOCValDataset(path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=40, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    polygon_metric = VOC07PolygonMApMetric(class_names=dataset.CLASSES)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        # targets[:, 2:] = xywh2xyxy(targets[:, 2:])

        imgs = Variable(imgs.type(Tensor))

        with torch.no_grad():
            s = time.time()
            outputs = model(imgs)
            s = time.time()
            outputs = non_max_suppression2(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            # print('NMS: ', time.time() - s)
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

        bbox_preds = []
        prob_preds = []
        class_preds = []
        center_preds = []
        coef_preds = []
        gt_bboxes = []
        gt_points_x = []
        gt_points_y = []
        gt_label = []
        for inx, output in enumerate(outputs):
            bbox_preds.append(output[..., :4].data.cpu().numpy())
            prob_preds.append(output[..., 4].data.cpu().numpy())
            class_preds.append(output[..., 6].data.cpu().numpy())
            center_preds.append(output[..., 6:8].data.cpu().numpy())
            coef_preds.append(output[..., 8: 26].data.cpu().numpy())
            target = targets[targets[:, 0] == inx]
            gt_bboxes.append(target[:, 2:6].data.cpu().numpy())
            gt_points_x.append(target[:, 6:366].data.cpu().numpy())
            gt_points_y.append(target[:, 366: 726].data.cpu().numpy())
            gt_label.append(target[:, 1].data.cpu().numpy())
        # print("time: ", time.time() - s)
        # embed()
        polygon_metric.update(bbox_preds, center_preds, coef_preds,
                              class_preds, prob_preds, gt_bboxes, gt_points_x, gt_points_y, gt_label)
        # print(polygon_metric.get())
    # Concatenate sample statistics
    names, values = polygon_metric.get()
    print("[Polygon MAP]")
    for k, y in zip(names, values):
        print(k, y)
    print('-------------')
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3_ese_seg.cfg", help="path to model definition file")
    parser.add_argument("--root", type=str, default="/workspace/cth/data/", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_0.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=opt.root,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({SBDVOCValDataset.CLASSES[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
