import importlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
from torch import multiprocessing, cuda
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from wsl_survey.segmentation.irn.misc import torchutils, imutils, pyutils, \
    indexing
from wsl_survey.segmentation.irn.voc12 import dataloader

cudnn.enabled = True

use_gpu = torch.cuda.is_available()


def find_centroids_with_refinement(displacement, iterations=300):
    # iteration: the number of refinement steps (u), set to any integer >= 100.

    height, width = displacement.shape[1:3]

    # 1. initialize centroids as their coordinates
    centroid_y = np.repeat(np.expand_dims(np.arange(height), 1), width,
                           axis=1).astype(np.float32)
    centroid_x = np.repeat(np.expand_dims(np.arange(width), 0), height,
                           axis=0).astype(np.float32)

    for i in range(iterations):
        # 2. find numbers after the decimals
        uy = np.ceil(centroid_y).astype(np.int32)
        dy = np.floor(centroid_y).astype(np.int32)
        y_c = centroid_y - dy

        ux = np.ceil(centroid_x).astype(np.int32)
        dx = np.floor(centroid_x).astype(np.int32)
        x_c = centroid_x - dx

        # 3. move centroids
        centroid_y += displacement[0][uy, ux] * y_c * x_c + \
                      displacement[0][dy, ux] * (1 - y_c) * x_c + \
                      displacement[0][uy, dx] * y_c * (1 - x_c) + \
                      displacement[0][dy, dx] * (1 - y_c) * (1 - x_c)

        centroid_x += displacement[1][uy, ux] * y_c * x_c + \
                      displacement[1][dy, ux] * (1 - y_c) * x_c + \
                      displacement[1][uy, dx] * y_c * (1 - x_c) + \
                      displacement[1][dy, dx] * (1 - y_c) * (1 - x_c)

        # 4. bound centroids
        centroid_y = np.clip(centroid_y, 0, height - 1)
        centroid_x = np.clip(centroid_x, 0, width - 1)

    centroid_y = np.round(centroid_y).astype(np.int32)
    centroid_x = np.round(centroid_x).astype(np.int32)

    return np.stack([centroid_y, centroid_x], axis=0)


def cluster_centroids(centroids, displacement, thres=2.5):
    # thres: threshold for grouping centroid (see supp)

    dp_strength = np.sqrt(displacement[1] ** 2 + displacement[0] ** 2)
    height, width = dp_strength.shape

    weak_dp_region = dp_strength < thres

    dp_label = measure.label(weak_dp_region, connectivity=1, background=0)
    dp_label_1d = dp_label.reshape(-1)

    centroids_1d = centroids[0] * width + centroids[1]

    clusters_1d = dp_label_1d[centroids_1d]

    cluster_map = imutils.compress_range(
        clusters_1d.reshape(height, width) + 1)

    return pyutils.to_one_hot(cluster_map)


def separte_score_by_mask(scores, masks):
    instacne_map_expanded = torch.from_numpy(
        np.expand_dims(masks, 0).astype(np.float32))
    if use_gpu:
        instance_score = torch.unsqueeze(scores,
                                         1) * instacne_map_expanded.cuda()
    else:
        instance_score = torch.unsqueeze(scores, 1) * instacne_map_expanded
    return instance_score


def detect_instance(score_map, mask, class_id, max_fragment_size=0):
    # converting pixel-wise instance ids into detection form

    pred_score = []
    pred_label = []
    pred_mask = []

    for ag_score, ag_mask, ag_class in zip(score_map, mask, class_id):
        if np.sum(ag_mask) < 1:
            continue
        segments = pyutils.to_one_hot(
            measure.label(ag_mask, connectivity=1, background=0))[1:]
        # connected components analysis

        for seg_mask in segments:
            if np.sum(seg_mask) < max_fragment_size:
                pred_score.append(0)
            else:
                pred_score.append(np.max(ag_score * seg_mask))
            pred_label.append(ag_class)
            pred_mask.append(seg_mask)

    return {
        'score': np.stack(pred_score, 0),
        'mask': np.stack(pred_mask, 0),
        'class': np.stack(pred_label, 0)
    }


def _work_cpu(process_id, model, dataset, args):
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False,
                             num_workers=1,
                             pin_memory=False)

    with torch.no_grad():

        for iter, pack in tqdm(enumerate(data_loader), total=len(databin)):
            img_name = pack['name'][0]
            path = os.path.join(args.ins_seg_out_dir, img_name + '.npy')
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                size = np.asarray(pack['size'])

                edge, dp = model(pack['img'][0])

                dp = dp.cpu().numpy()

                cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy',
                                   allow_pickle=True).item()

                cams = cam_dict['cam']
                keys = cam_dict['keys']

                centroids = find_centroids_with_refinement(dp)
                instance_map = cluster_centroids(centroids, dp)
                instance_cam = separte_score_by_mask(cams, instance_map)

                rw = indexing.propagate_to_edge(instance_cam,
                                                edge,
                                                beta=args.beta,
                                                exp_times=args.exp_times,
                                                radius=5)

                rw_up = F.interpolate(rw,
                                      scale_factor=4,
                                      mode='bilinear',
                                      align_corners=False)[:,
                        0, :size[0], :size[1]]
                rw_up = rw_up / torch.max(rw_up)

                rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0),
                                 value=args.ins_seg_bg_thres)

                num_classes = len(keys)
                num_instances = instance_map.shape[0]

                instance_shape = torch.argmax(rw_up_bg, 0).cpu().numpy()
                instance_shape = pyutils.to_one_hot(
                    instance_shape,
                    maximum_val=num_instances * num_classes + 1)[1:]
                instance_class_id = np.repeat(keys, num_instances)

                detected = detect_instance(rw_up.cpu().numpy(),
                                           instance_shape,
                                           instance_class_id,
                                           max_fragment_size=size[0] * size[1] *
                                                             0.01)

                np.save(path,
                        detected)

                if process_id == args.num_workers - 1 and iter % (len(databin) //
                                                                  4) == 0:
                    print("%d " % ((5 * iter + 1) // (len(databin) // 4)), end='')


def _work_gpu(process_id, model, dataset, args):
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False,
                             num_workers=args.num_workers // n_gpus,
                             pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in tqdm(enumerate(data_loader), total=len(databin)):
            img_name = pack['name'][0]
            path = os.path.join(args.ins_seg_out_dir, img_name + '.npy')
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                size = np.asarray(pack['size'])

                edge, dp = model(pack['img'][0].cuda(non_blocking=True))

                dp = dp.cpu().numpy()

                cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy',
                                   allow_pickle=True).item()

                cams = cam_dict['cam'].cuda()
                keys = cam_dict['keys']

                centroids = find_centroids_with_refinement(dp)
                instance_map = cluster_centroids(centroids, dp)
                instance_cam = separte_score_by_mask(cams, instance_map)

                rw = indexing.propagate_to_edge(instance_cam,
                                                edge,
                                                beta=args.beta,
                                                exp_times=args.exp_times,
                                                radius=5)

                rw_up = F.interpolate(rw,
                                      scale_factor=4,
                                      mode='bilinear',
                                      align_corners=False)[:,
                        0, :size[0], :size[1]]
                rw_up = rw_up / torch.max(rw_up)

                rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0),
                                 value=args.ins_seg_bg_thres)

                num_classes = len(keys)
                num_instances = instance_map.shape[0]

                instance_shape = torch.argmax(rw_up_bg, 0).cpu().numpy()
                instance_shape = pyutils.to_one_hot(
                    instance_shape,
                    maximum_val=num_instances * num_classes + 1)[1:]
                instance_class_id = np.repeat(keys, num_instances)

                detected = detect_instance(rw_up.cpu().numpy(),
                                           instance_shape,
                                           instance_class_id,
                                           max_fragment_size=size[0] * size[1] *
                                                             0.01)

                np.save(path,
                        detected)

                if process_id == n_gpus - 1 and iter % (len(databin) // 4) == 0:
                    print("%d " % ((5 * iter + 1) // (len(databin) // 4)), end='')


def run(args):
    assert args.voc12_root is not None
    assert args.class_label_dict_path is not None
    assert args.infer_list is not None
    assert args.ins_seg_out_dir is not None
    assert args.irn_weights_name is not None
    assert args.cam_out_dir is not None
    assert args.irn_network is not None
    assert args.irn_network_module is not None

    model = getattr(importlib.import_module(args.irn_network_module),
                    args.irn_network + 'EdgeDisplacement')()

    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
    model.eval()
    dataset = dataloader.VOC12ClassificationDatasetMSF(
        args.infer_list,
        voc12_root=args.voc12_root,
        scales=(1.0,),
        class_label_dict_path=args.class_label_dict_path)

    if use_gpu:
        n_gpus = torch.cuda.device_count()

        dataset = torchutils.split_dataset(dataset, n_gpus)
        multiprocessing.spawn(_work_gpu,
                              nprocs=n_gpus,
                              args=(model, dataset, args),
                              join=True)
    else:
        dataset = torchutils.split_dataset(dataset, args.num_workers)
        multiprocessing.spawn(_work_cpu,
                              nprocs=args.num_workers,
                              args=(model, dataset, args),
                              join=True)
    print("[ ", end='')

    print("]")


if __name__ == '__main__':
    from wsl_survey.segmentation.irn.config import make_parser

    parser = make_parser()
    parser.set_defaults(
        voc12_root='./data/test1/VOC2012',
        class_label_dict_path='./data/voc12/cls_labels.npy',
        infer_list='./data/test1/VOC2012/ImageSets/Segmentation/val.txt',
        irn_network='ResNet18',
        num_workers=1,
        ins_seg_out_dir='./outputs/test1/results/resnet18/ins_seg',
        irn_weights_name='./outputs/test1/results/resnet18/sess/irn.pth',
        cam_out_dir='./outputs/test1/results/resnet18/cam',
        irn_network_module='wsl_survey.segmentation.irn.net.resnet_irn',
    )
    args = parser.parse_args()
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)
    run(args)
