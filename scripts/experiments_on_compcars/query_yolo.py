import json
import os
import urllib.request

from tqdm import tqdm


def parse_path(x):
    x = x.strip()
    image_path = os.path.join('file:///data/image', x)
    return x, image_path


if __name__ == '__main__':
    # results = []
    #
    # with open('./data/compcars/arxiv_data/train_test_split/classification_train.txt', mode='r') as f:
    #     for line in tqdm(f.readlines()):
    #         img_name, image_path = parse_path(line)
    #         try:
    #             resp = urllib.request.urlopen(
    #                 "http://localhost:80/api?image_url=%s" % image_path).read().decode(
    #                 'utf-8')
    #             results.append({img_name: resp})
    #         except Exception as e:
    #             print(e)
    # os.makedirs('/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/yolo/', exist_ok=True)
    # with open('/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/yolo/tr_results.txt', mode='w') as f:
    #     for i in results:
    #         f.write('%s\n' % json.dumps(i))

    results = []
    counter = 0
    result_path = './compcars_outputs/compcars/yolo/te_results.txt'
    image_names = []
    try:
        with open(result_path, mode='r') as f:
            for i in  f.readlines():
                image_names.extend(json.loads(i).keys())
    except:
        pass
    os.makedirs('./compcars_outputs/compcars/yolo/', exist_ok=True)
    with open('./data/compcars/arxiv_data/train_test_split/classification_test.txt', mode='r') as f:
        for line in tqdm(f.readlines()):
            img_name, image_path = parse_path(line)
            if not img_name in image_names:
                try:
                    resp = urllib.request.urlopen(
                        "http://localhost:80/api?image_url=%s" % image_path).read().decode(
                        'utf-8')

                    results.append({img_name: resp})
                except Exception as e:
                    print(e)
            counter += 1
            if counter % 100 == 0:
                with open(result_path,
                          mode='w') as f:
                    for i in results:
                        f.write('%s\n' % json.dumps(i))
    with open('./compcars_outputs/compcars/yolo/te_results.txt', mode='w') as f:
        for i in results:
            f.write('%s\n' % json.dumps(i))
