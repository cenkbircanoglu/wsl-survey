import os

import numpy as np
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from tqdm import tqdm


def execute_rules(args, label):
    """
    cat_list = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    subsets = {
        'subset1': ['cat', 'dog'],
        'subset2': ['bus', 'car'],
        'subset3': ['cat', 'dog', 'horse'],
        'subset4': ['bus', 'car', 'train'],
        'subset5': ['cat', 'dog', 'bus'],
        'subset6': ['cat', 'dog', 'horse', 'bus'],
        'subset7': ['cat', 'bus', 'car', 'train'],
        'subset8': ['cat', 'dog', 'horse', 'bus', 'car'],
        'subset9': ['cat', 'dog', 'bus', 'car', 'train'],
        'subset10': ['cat', 'dog', 'horse', 'bus', 'car', 'train']
    }
    """
    if 'subset10' in args.cam_out_dir:
        label[(label != 8) & (label != 12) & (label != 13) & (label != 6) & (label != 7) & (label != 19)] = 0
        label[label == 8] = 1
        label[label == 12] = 2
        label[label == 13] = 3
        label[label == 6] = 4
        label[label == 7] = 5
        label[label == 19] = 6
    elif 'subset1' in args.cam_out_dir:
        label[(label != 8) & (label != 12)] = 0
        label[label == 8] = 1
        label[label == 12] = 2
    elif 'subset2' in args.cam_out_dir:
        label[(label != 6) & (label != 7)] = 0
        label[label == 6] = 1
        label[label == 7] = 2
    elif 'subset3' in args.cam_out_dir:
        label[(label != 8) & (label != 12) & (label != 13)] = 0
        label[label == 8] = 1
        label[label == 12] = 2
        label[label == 13] = 3
    elif 'subset4' in args.cam_out_dir:
        label[(label != 6) & (label != 7) & (label != 19)] = 0
        label[label == 6] = 1
        label[label == 7] = 2
        label[label == 19] = 3
    elif 'subset5' in args.cam_out_dir:
        label[(label != 8) & (label != 12) & (label != 6)] = 0
        label[label == 8] = 1
        label[label == 12] = 2
        label[label == 6] = 3
    elif 'subset6' in args.cam_out_dir:
        label[(label != 8) & (label != 12) & (label != 13) & (label != 6)] = 0
        label[label == 8] = 1
        label[label == 12] = 2
        label[label == 13] = 3
        label[label == 6] = 4
    elif 'subset7' in args.cam_out_dir:
        label[(label != 8) & (label != 6) & (label != 7) & (label != 19)] = 0
        label[label == 8] = 1
        label[label == 6] = 2
        label[label == 7] = 3
        label[label == 19] = 4
    elif 'subset8' in args.cam_out_dir:
        label[(label != 8) & (label != 12) & (label != 13) & (label != 6) & (label != 7)] = 0
        label[label == 8] = 1
        label[label == 12] = 2
        label[label == 13] = 3
        label[label == 6] = 4
        label[label == 7] = 5
    elif 'subset9' in args.cam_out_dir:
        label[(label != 8) & (label != 12) & (label != 6) & (label != 7) & (label != 19)] = 0
        label[label == 8] = 1
        label[label == 12] = 2
        label[label == 6] = 3
        label[label == 7] = 4
        label[label == 19] = 5

    return label


def run(args):
    assert args.voc12_root is not None
    assert args.chainer_eval_set is not None
    assert args.cam_out_dir is not None

    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set,
                                             data_dir=args.voc12_root)

    preds = []
    labels = []
    for d, id in tqdm(enumerate(dataset.ids)):
        try:
            cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'),
                               allow_pickle=True).item()
            cams = cam_dict['high_res']
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)),
                          mode='constant',
                          constant_values=args.cam_eval_thres)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            pred = cls_labels.copy()
            label = dataset.get_example_by_keys(d, (1,))[0]
            label = execute_rules(args, label)
            preds.append(pred)
            labels.append(label)
        except Exception as e:
            pass
    confusion = calc_semantic_segmentation_confusion(preds, labels)
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    print(args.cam_out_dir)
    print({'iou': iou, 'miou': np.nanmean(iou)})


if __name__ == '__main__':
    from wsl_survey.segmentation.irn.config import make_parser

    parser = make_parser()
    parser.set_defaults(voc12_root='./datasets/voc2012/VOCdevkit/VOC2012',
                        chainer_eval_set='train',
                        cam_out_dir='./results/subset9_resnet152/subset9_resnet152/cam')
    args = parser.parse_args()
    run(args)
