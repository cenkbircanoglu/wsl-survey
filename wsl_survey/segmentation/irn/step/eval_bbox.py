import os
from statistics import mean

from tqdm import tqdm

from wsl_survey.segmentation.irn.voc12 import dataloader


def bb_intersection_over_union(boxA, boxB):
    boxA = [int(i) for i in boxA]
    boxB = [int(i) for i in boxB]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def run(args):
    assert args.voc12_root is not None

    dataset = dataloader.VOC12ImageDataset(args.infer_list,
                                           voc12_root=args.voc12_root,
                                           img_normal=None,
                                           to_torch=False)

    preds = []
    for data in tqdm(dataset):
        img_name = data['name']
        bbox_org_path = os.path.join(args.voc12_root, img_name.replace('image', 'label') + '.txt')
        with open(bbox_org_path, mode='r') as f:
            bbox_org = f.readlines()[-1].strip().split(' ')
        bbox_path = os.path.join(args.bbox_out_dir, img_name + '.txt')
        with open(bbox_path, mode='r') as f:
            bbox = f.readlines()[0].strip().split('\t')
        iou = bb_intersection_over_union(bbox, bbox_org)
        preds.append(iou)
    print({'miou': mean(preds)})


if __name__ == '__main__':
    from wsl_survey.segmentation.irn.config import make_parser

    parser = make_parser()
    parser.set_defaults(voc12_root='./data/compcars/',
                        chainer_eval_set='./data/compcars/train/test.txt',
                        cam_out_dir='./outputs/test1/results/resnet18/cam')
    args = parser.parse_args()
    run(args)
