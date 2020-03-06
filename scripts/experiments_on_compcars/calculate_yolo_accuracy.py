import json
import os
from statistics import mean

from tqdm import tqdm

from wsl_survey.segmentation.irn.voc12 import dataloader

yolo_result_path = '/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/yolo/bbox'
with open('compcars_outputs/compcars/yolo/te_results.txt', mode='r') as f:
    for i in f.readlines():
        item = json.loads(i)
        max_area = -1
        for key, predictions in item.items():
            predictions = eval(predictions)
            for prediction in predictions:
                if prediction['category'] in ['truck', 'car']:
                    bbox = prediction['bbox']
                    x, y, w, h = bbox
                    area = w * h

                    if area > max_area:
                        max_area = area
                        prediction = bbox
            path = os.path.join(yolo_result_path, key.replace('.jpg', '.txt'))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, mode='w') as f:
                x, y, w, h = bbox
                f.write('%s\t%s\t%s\t%s\n' % (x, y, w, h))


def bb_intersection_over_union(boxA, boxB):
    boxA = [int(float(i)) for i in boxA]
    x, y, w, h = boxA
    boxA[0] = int(x - w / 2)
    boxA[1] = int(y - h / 2)
    boxA[2] = int(x + w / 2)
    boxA[3] = int(y + h / 2)

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


def run():
    voc12_root = './data/compcars/'
    bbox_out_dir = yolo_result_path
    dataset = dataloader.VOC12ImageDataset('./data/compcars/train/test.txt',
                                           voc12_root=voc12_root,
                                           img_normal=None,
                                           to_torch=False)

    preds = []
    error_cnt = 0
    for data in tqdm(dataset):
        img_name = data['name']
        bbox_org_path = os.path.join(voc12_root, img_name.replace('image', 'label') + '.txt')
        try:
            with open(bbox_org_path, mode='r') as f:
                bbox_org = f.readlines()[-1].strip().split(' ')
            bbox_path = os.path.join(bbox_out_dir, img_name + '.txt').replace('data/image/', '')
            with open(bbox_path, mode='r') as f:
                bbox = f.readlines()[0].strip().split('\t')
            iou = bb_intersection_over_union(bbox, bbox_org)
            preds.append(iou)
        except Exception as e:
            error_cnt += 1

    print({'miou': mean(preds)}, len(preds), error_cnt, bbox_out_dir)


if __name__ == '__main__':
    run()
