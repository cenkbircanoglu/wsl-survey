import os

import chainercv
import numpy as np
from chainercv.datasets import VOCInstanceSegmentationDataset
from tqdm import tqdm


def run(args):
    assert args.voc12_root is not None
    assert args.chainer_eval_set is not None
    assert args.ins_seg_out_dir is not None

    dataset = VOCInstanceSegmentationDataset(split=args.chainer_eval_set,
                                             data_dir=args.voc12_root)
    gt_masks = [
        dataset.get_example_by_keys(i, (1, ))[0] for i in range(len(dataset))
    ]
    gt_labels = [
        dataset.get_example_by_keys(i, (2, ))[0] for i in range(len(dataset))
    ]

    pred_class = []
    pred_mask = []
    pred_score = []
    for id in tqdm(dataset.ids):
        ins_out = np.load(os.path.join(args.ins_seg_out_dir, id + '.npy'),
                          allow_pickle=True).item()
        pred_class.append(ins_out['class'])
        pred_mask.append(ins_out['mask'])
        pred_score.append(ins_out['score'])
    print(
        '0.5iou:',
        chainercv.evaluations.eval_instance_segmentation_voc(pred_mask,
                                                             pred_class,
                                                             pred_score,
                                                             gt_masks,
                                                             gt_labels,
                                                             iou_thresh=0.5))
    print(
        '0.7iou:',
        chainercv.evaluations.eval_instance_segmentation_voc(pred_mask,
                                                             pred_class,
                                                             pred_score,
                                                             gt_masks,
                                                             gt_labels,
                                                             iou_thresh=0.7))


if __name__ == '__main__':
    from wsl_survey.segmentation.irn.config import make_parser

    parser = make_parser()
    parser.set_defaults(
        voc12_root='./data/test1/VOC2012',
        chainer_eval_set='val',
        ins_seg_out_dir='./outputs/test1/results/resnet18/ins_seg',
    )
    args = parser.parse_args()
    run(args)
