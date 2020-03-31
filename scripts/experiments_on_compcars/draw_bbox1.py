import os

import cv2
from tqdm import tqdm

choices = []
with open('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars/arxiv_data/train_test_split/classification_test.txt',
          mode='r') as f:
    lines = f.readlines()
    for i in lines:
        choices.append(i)
for value in tqdm(choices):
    value = value.strip()
    path = os.path.join('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars/data/image', value)
    result_path = '/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/results_1/'
    gt_path_bbox = os.path.join('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars/data/label/',
                                value.replace('jpg', 'txt'))
    model_path_bbox = os.path.join(
        '/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/model/results/resnet50/bbox/data/image/',
        value.replace('jpg', 'txt'))
    kmeans431_path_bbox = os.path.join(
        '/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/kmeans_431/results/resnet50/bbox/data/image/',
        value.replace('jpg', 'txt'))
    yolo_path_bbox = os.path.join(
        '/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/yolo/bbox',
        value.replace('jpg', 'txt')).replace('data/image/', '')

    im = cv2.imread(path)
    try:
        with open(gt_path_bbox, mode='r') as f:
            bbox = f.readlines()[2].split(' ')
            x, y, w, h = map(int, bbox)
            color_code = (255, 0, 0)
            text = 'Original'
            cv2.rectangle(im, (x, y), (w, h), color_code, 1)
            cv2.putText(im, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_code, 2)

        with open(yolo_path_bbox, mode='r') as f:
            bbox = f.readlines()[0].strip().split('\t')
            x, y, w, h = map(lambda x: int(float(x)), bbox)
            color_code = (0, 255, 0)
            text = 'Yolo'
            cv2.rectangle(im, (x, y), (x + w, h + y), color_code, 1)
            cv2.putText(im, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_code, 2)

        with open(model_path_bbox, mode='r') as f:
            bbox = f.readlines()[0].strip().split('\t')
            x, y, w, h = map(int, bbox)
            color_code = (0, 0, 255)
            text = 'Model'
            cv2.rectangle(im, (x, y), (x + w, h + y), color_code, 1)
            cv2.putText(im, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_code, 2)

        with open(kmeans431_path_bbox, mode='r') as f:
            bbox = f.readlines()[0].strip().split('\t')
            x, y, w, h = map(int, bbox)
            color_code = (255, 255, 0)
            text = 'KMeans 431'
            cv2.rectangle(im, (x, y), (x + w, h + y), color_code, 1)
            cv2.putText(im, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_code, 2)

        # cv2.imshow("Show", im)
        # cv2.waitKey()
        output_value = value.replace('/', '_')
        path = os.path.join(result_path, output_value)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, im)
    except:
        pass
