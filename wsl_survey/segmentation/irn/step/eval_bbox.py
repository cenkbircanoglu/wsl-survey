import os
from statistics import mean

from tqdm import tqdm
import numpy as np
from wsl_survey.segmentation.irn.voc12 import dataloader
from wsl_survey.utils.iou import bb_intersection_over_union


def run(args):
    assert args.voc12_root is not None

    dataset = dataloader.VOC12ImageDataset(args.infer_list,
                                           voc12_root=args.voc12_root,
                                           img_normal=None,
                                           to_torch=False)

    preds = []
    preds_dict = {}
    error_cnt = 0
    label_dict = np.load(args.class_label_dict_path, allow_pickle=True).item()
    for data in tqdm(dataset):
        img_name = data['name']
        label = np.argmax(label_dict[img_name])
        bbox_org_path = os.path.join(args.voc12_root, img_name.replace('image', 'label') + '.txt')
        try:
            with open(bbox_org_path, mode='r') as f:
                bbox_org = f.readlines()[-1].strip().split(' ')
            bbox_path = os.path.join(args.bbox_out_dir, img_name + '.txt')
            with open(bbox_path, mode='r') as f:
                bbox = f.readlines()[0].strip().split('\t')
            iou = bb_intersection_over_union(bbox, bbox_org)
            preds.append(iou)
            preds_dict.setdefault(label, []).append(iou)
        except Exception as e:
            error_cnt += 1

    print({'miou': mean(preds)}, len(preds), error_cnt, args.bbox_out_dir)
    print({key: mean(value) for key, value in preds_dict.items()})


if __name__ == '__main__':
    from wsl_survey.segmentation.irn.config import make_parser

    parser = make_parser()
    parser.set_defaults(voc12_root='./data/compcars/',
                        chainer_eval_set='./data/compcars/train/test.txt',
                        cam_out_dir='./outputs/test1/results/resnet18/cam')
    args = parser.parse_args()
    run(args)
