import os

import numpy as np
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from tqdm import tqdm


def run(args):
    assert args.voc12_root is not None
    assert args.chainer_eval_set is not None
    assert args.cam_out_dir is not None

    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set,
                                             data_dir=args.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in
              range(len(dataset))]

    preds = []
    for id in tqdm(dataset.ids):
        cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'),
                           allow_pickle=True).item()
        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant',
                      constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())
    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    print({'iou': iou, 'miou': np.nanmean(iou)})


if __name__ == '__main__':
    from wsl_survey.segmentation.irn.config import make_parser

    parser = make_parser()
    parser.set_defaults(
        voc12_root='./data/test/VOC2012',
        chainer_eval_set='train',
        cam_out_dir='./outputs/test/results/resnet18/cam'
    )
    args = parser.parse_args()
    run(args)
