import os

import chainercv
import numpy as np
from chainercv.datasets import VOCInstanceSegmentationDataset
from tqdm import tqdm


def run(args):
    dataset = VOCInstanceSegmentationDataset(split=args.chainer_eval_set,
                                             data_dir=args.voc12_root)
    gt_masks = [dataset.get_example_by_keys(i, (1,))[0] for i in
                range(len(dataset))]
    gt_labels = [dataset.get_example_by_keys(i, (2,))[0] for i in
                 range(len(dataset))]

    pred_class = []
    pred_mask = []
    pred_score = []
    for id in tqdm(dataset.ids):
        ins_out = np.load(os.path.join(args.ins_seg_out_dir, id + '.npy'),
                          allow_pickle=True).item()
        pred_class.append(ins_out['class'])
        pred_mask.append(ins_out['mask'])
        pred_score.append(ins_out['score'])
    print('0.5iou:',
          chainercv.evaluations.eval_instance_segmentation_voc(pred_mask,
                                                               pred_class,
                                                               pred_score,
                                                               gt_masks,
                                                               gt_labels,
                                                               iou_thresh=0.5))
    print('0.7iou:',
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
