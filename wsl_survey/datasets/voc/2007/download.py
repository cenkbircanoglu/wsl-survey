import os
import os.path
import shutil
import tarfile

from wsl_survey.datasets.utils import download_if_not_exists

urls = {
    'trval':
    'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'test':
    'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'
}


def untar_file(file, output):
    os.makedirs(output, exist_ok=True)
    print('[dataset] Extracting tar file {file} to {path}'.format(file=file,
                                                                  path=output))
    cwd = os.getcwd()
    tar = tarfile.open(file, "r")
    os.chdir(output)
    tar.extractall()
    tar.close()
    os.chdir(cwd)
    print('[dataset] Done!')


def download(output):
    output = os.path.abspath(output)
    os.makedirs(output, exist_ok=True)
    tmp_dir = os.path.join(output, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    for data_type in ['trval', 'test']:
        filename, cached_file = download_if_not_exists(data_type, tmp_dir,
                                                       urls)
        untar_file(cached_file, output)

    file_list = [
        'VOCdevkit/VOC2007/Annotations/000001.xml',
        'VOCdevkit/VOC2007/ImageSets/Layout/test.txt',
        'VOCdevkit/VOC2007/ImageSets/Layout/train.txt',
        'VOCdevkit/VOC2007/ImageSets/Layout/trainval.txt',
        'VOCdevkit/VOC2007/ImageSets/Layout/val.txt',
        'VOCdevkit/VOC2007/ImageSets/Main/aeroplane_test.txt',
        'VOCdevkit/VOC2007/ImageSets/Main/aeroplane_train.txt',
        'VOCdevkit/VOC2007/ImageSets/Main/aeroplane_trainval.txt',
        'VOCdevkit/VOC2007/ImageSets/Main/aeroplane_val.txt',
        'VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt',
        'VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt',
        'VOCdevkit/VOC2007/ImageSets/Segmentation/trainval.txt',
        'VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt',
        'VOCdevkit/VOC2007/JPEGImages/000001.jpg',
        'VOCdevkit/VOC2007/SegmentationClass/000032.png'
    ]
    if any(
            map(lambda x: not os.path.exists(os.path.join(output, x)),
                file_list)):
        download(output)
    data_dir = os.path.join(output, 'data')

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)

    source = os.path.join(output, 'VOCdevkit', 'VOC2007')

    for f in [
            'Annotations', 'ImageSets', 'JPEGImages', 'SegmentationClass',
            'SegmentationObject'
    ]:
        shutil.copytree(os.path.join(source, f), os.path.join(data_dir, f))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Pascal VOC2007 Dataset Downloader')
    parser.add_argument('--dataset_dir',
                        metavar='DIR',
                        help='path to data dir')
    args = parser.parse_args()
    download(args.dataset_dir)
