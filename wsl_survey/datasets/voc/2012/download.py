import os
import os.path
import shutil
import tarfile

from wsl_survey.datasets.utils import download_if_not_exists

urls = {
    'trval':
    'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
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

    for data_type in ['trval']:
        filename, cached_file = download_if_not_exists(data_type, tmp_dir,
                                                       urls)
        untar_file(cached_file, output)

    file_list = [
        'VOCdevkit/VOC2012/Annotations/2007_000027.xml',
        'VOCdevkit/VOC2012/ImageSets/Layout/train.txt',
        'VOCdevkit/VOC2012/ImageSets/Layout/trainval.txt',
        'VOCdevkit/VOC2012/ImageSets/Layout/val.txt',
        'VOCdevkit/VOC2012/ImageSets/Main/aeroplane_train.txt',
        'VOCdevkit/VOC2012/ImageSets/Main/aeroplane_trainval.txt',
        'VOCdevkit/VOC2012/ImageSets/Main/aeroplane_val.txt',
        'VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt',
        'VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt',
        'VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt',
        'VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg',
        'VOCdevkit/VOC2012/SegmentationClass/2007_000032.png'
    ]
    if any(
            map(lambda x: not os.path.exists(os.path.join(output, x)),
                file_list)):
        download(output)
    data_dir = os.path.join(output, 'data')
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)

    source = os.path.join(output, 'VOCdevkit', 'VOC2012')

    for f in [
            'Annotations', 'ImageSets', 'JPEGImages', 'SegmentationClass',
            'SegmentationObject'
    ]:
        shutil.copytree(os.path.join(source, f), os.path.join(data_dir, f))
    # shutil.rmtree(os.path.join(output, 'VOCdevkit'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Pascal VOC2007 Dataset Downloader')
    parser.add_argument('--dataset_dir',
                        metavar='DIR',
                        help='path to data dir')
    args = parser.parse_args()
    download(args.dataset_dir)
