import json
import os
from statistics import mean

from tqdm import tqdm

from wsl_survey.segmentation.irn.voc12 import dataloader
from wsl_survey.utils.iou import bb_intersection_over_union

yolo_result_path = '/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/yolo/bbox'
category_list = set()
with open('compcars_outputs/compcars/yolo/te_results.txt', mode='r') as f:
    for i in f.readlines():
        item = json.loads(i)

        for key, predictions in item.items():
            max_area = -1
            predictions = eval(predictions)
            best_prediction = None
            for prediction in predictions:
                category_list.add(prediction['category'])
                if prediction['category'] in ['truck', 'car', 'bus']:
                    bbox = prediction['bbox']
                    x, y, w, h = bbox
                    area = w * h
                    if area > max_area:
                        max_area = area
                        best_prediction = (int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2))
            path = os.path.join(yolo_result_path, key.replace('.jpg', '.txt'))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, mode='w') as f:
                try:
                    x1, y1, w1, h1 = best_prediction
                    w1 = w1 -x1
                    h1 = h1 - y1
                except:
                    pass
                f.write('%s\t%s\t%s\t%s\n' % (x1, y1, w1, h1))


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
            print(e)
            error_cnt += 1

    print({'miou': mean(preds)}, len(preds), error_cnt, bbox_out_dir)


if __name__ == '__main__':
    run()
