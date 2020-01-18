import os

import imageio
import numpy as np
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from tqdm import tqdm


def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set,
                                             data_dir=args.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in
              range(len(dataset))]

    preds = []
    for id in tqdm(dataset.ids):
        cls_labels = imageio.imread(
            os.path.join(args.sem_seg_out_dir, id + '.png')).astype(np.uint8)
        cls_labels[cls_labels == 255] = 0
        preds.append(cls_labels.copy())
    confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator

    print(fp[0], fn[0])
    print(np.mean(fp[1:]), np.mean(fn[1:]))

    print({'iou': iou, 'miou': np.nanmean(iou)})

if __name__ == '__main__':
    from wsl_survey.segmentation.irn.config import make_parser

    parser = make_parser()
    parser.set_defaults(
        voc12_root='./data/test/VOC2012',
        class_label_dict_path='./data/test/VOC2012/ImageSets/Segmentation/cls_labels.npy',
        train_list='./data/test/VOC2012/ImageSets/Segmentation/train_aug.txt',
        ir_label_out_dir='./outputs/test/results/resnet18/irn_label',
        infer_list='./data/voc12/train.txt',
        irn_network='ResNet18',
        irn_num_epoches=1,
        irn_batch_size=4
    )
    args = parser.parse_args()
    run(args)
