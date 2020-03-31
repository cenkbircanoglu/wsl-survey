import os

import cv2
import numpy as np
import tqdm


def generate_bbox(path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mask_img = cv2.imread(path)
    ret, threshed_img = cv2.threshold(cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(threshed_img, cv2.MORPH_CLOSE, kernel, iterations=8)

    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest_area = float('-inf')
    x, y, w, h = None, None, None, None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > biggest_area:
            biggest_area = area
            x, y, w, h = cv2.boundingRect(cnt)
    with open(output_path, mode='w') as f:
        f.write('%s\t%s\t%s\t%s\n' % (x, y, w, h))


with open('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars/data/train_test_split/classification/test.txt',
          mode='r') as f:
    paths = f.readlines()

image_folder = '/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars/data/image'
output_folder = '/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/cv_results'
for path in tqdm.tqdm(paths):
    path = path.strip()
    input_path = os.path.join(image_folder, path)
    output_path = os.path.join(output_folder, path.replace('.jpg', '.txt'))
    generate_bbox(input_path, output_path)
