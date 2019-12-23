import os
import os.path
import shutil

from wsl_survey.datasets.utils import download_if_not_exists, unzip_file

urls = {
    'trainval_anno':
    'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    'test': 'http://images.cocodataset.org/zips/test2014.zip',
    'val': 'http://images.cocodataset.org/zips/val2014.zip',
    'train': 'http://images.cocodataset.org/zips/train2014.zip',
}


def download(output):
    os.makedirs(output, exist_ok=True)
    tmp_dir = os.path.join(output, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    trainval_anno_filename, trainval_anno_cached_file = download_if_not_exists(
        'trainval_anno', tmp_dir, urls)
    unzip_file(trainval_anno_cached_file, output)

    test_filename, test_cached_file = download_if_not_exists(
        'test', tmp_dir, urls)
    unzip_file(test_cached_file, output)

    val_filename, val_cached_file = download_if_not_exists(
        'val', tmp_dir, urls)
    unzip_file(val_cached_file, output)

    train_filename, train_cached_file = download_if_not_exists(
        'train', tmp_dir, urls)
    unzip_file(train_cached_file, output)

    image_folder = os.path.join(output, 'images')
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)
    os.mkdir(image_folder)

    for f in ['test2014', 'val2014', 'train2014']:
        folder = os.path.join(output, f)
        for i in os.listdir(folder):
            shutil.move(os.path.join(folder, i), image_folder)
        shutil.rmtree(folder)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='COCO 2014 Dataset Downloader')
    parser.add_argument('--dataset_dir',
                        metavar='DIR',
                        help='path to data dir')
    args = parser.parse_args()
    download(args.dataset_dir)
