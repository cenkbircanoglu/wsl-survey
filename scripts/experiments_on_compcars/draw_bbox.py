import os
import random

import cv2

from scripts.experiments_on_compcars import calculate_yolo_accuracy
from wsl_survey.segmentation.irn.step.eval_bbox import bb_intersection_over_union

choices = []
with open('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars/arxiv_data/train_test_split/classification_test.txt',
          mode='r') as f:
    lines = f.readlines()
    for _ in range(10):
        value = random.choice(lines)
        choices.append(value)

for value in choices:
    value = value.strip()
    value = '54/188/2013/57b643c9bf2b48.jpg'
    path = os.path.join('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars/data/image', value)
    gt_path_bbox = os.path.join('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars/data/label/',
                                value.replace('jpg', 'txt'))
    make_path_bbox = os.path.join(
        '/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/make/results/resnet50/bbox/data/image/',
        value.replace('jpg', 'txt'))
    model_path_bbox = os.path.join(
        '/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/model/results/resnet50/bbox/data/image/',
        value.replace('jpg', 'txt'))
    year_path_bbox = os.path.join(
        '/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/year/results/resnet50/bbox/data/image/',
        value.replace('jpg', 'txt'))
    make_year_path_bbox = os.path.join(
        '/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/make_year/results/resnet50/bbox/data/image/',
        value.replace('jpg', 'txt'))
    random_path_bbox = os.path.join(
        '/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/random_75/results/resnet50/bbox/data/image/',
        value.replace('jpg', 'txt'))
    yolo_path_bbox = os.path.join(
        '/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/yolo/bbox',
        value.replace('jpg', 'txt')).replace('data/image/', '')

    print(path)
    im = cv2.imread(path)

    with open(gt_path_bbox, mode='r') as f:
        bbox = f.readlines()[2].split(' ')
        x, y, w, h = map(int, bbox)
        boxb = (x, y, w, h)
        cv2.rectangle(im, (x, y), (w, h), (255, 0, 0), 1)
        cv2.putText(im, 'Ground Truth', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    with open(yolo_path_bbox, mode='r') as f:
        bbox = f.readlines()[0].strip().split('\t')
        x, y, w, h = map(lambda x: int(float(x)), bbox)
        boxa = (x, y,w,h)
        cv2.rectangle(im, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 0, 0), 1)
        cv2.putText(im, 'Yolo', (int(x - w / 2), int(y - h / 2) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    print('Yolo',calculate_yolo_accuracy.bb_intersection_over_union(boxa, boxb))
    with open(make_path_bbox, mode='r') as f:
        bbox = f.readlines()[0].strip().split('\t')
        x, y, w, h = map(int, bbox)
        boxa = (x, y, w, h)
        print(im.shape)
        print((x, y),( x+w,  y+h))
        cv2.rectangle(im, (x, y), (x+w, h+ y), (0, 255, 0), 1)
        cv2.putText(im, 'Make', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    print('Make', bb_intersection_over_union(boxa, boxb))
    with open(model_path_bbox, mode='r') as f:
        bbox = f.readlines()[0].strip().split('\t')
        x, y, w, h = map(int, bbox)
        boxa = (x, y, w, h)
        cv2.rectangle(im, (x, y), (w + x, h + y), (0, 0, 255), 2)
        cv2.putText(im, 'Model', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    print('Model', bb_intersection_over_union(boxa, boxb))
    with open(year_path_bbox, mode='r') as f:
        bbox = f.readlines()[0].strip().split('\t')
        x, y, w, h = map(int, bbox)
        cv2.rectangle(im, (x, y), (w + x, h + y), (255, 255, 0), 2)
        cv2.putText(im, 'Year', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    with open(make_year_path_bbox, mode='r') as f:
        bbox = f.readlines()[0].strip().split('\t')
        x, y, w, h = map(int, bbox)
        cv2.rectangle(im, (x, y), (w + x, h + y), (255, 0, 255), 2)
        cv2.putText(im, 'Make Year', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    try:
        with open(random_path_bbox, mode='r') as f:
            bbox = f.readlines()[0].strip().split('\t')
            x, y, w, h = map(int, bbox)
            cv2.rectangle(im, (x, y), (w + x, h + y), (0, 255, 255), 2)
            cv2.putText(im, 'Random 75', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    except:
        pass

    cv2.imshow("Show", im)
    cv2.waitKey()
