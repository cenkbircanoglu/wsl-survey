import os
import os.path

from wsl_survey.datasets.utils import download_if_not_exists, unzip_file

urls = {
    'data':
    'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data_downloader.zip'
}


def run_sh(file, output):
    if not os.path.exists(output):
        os.makedirs(output)
    cwd = os.getcwd()
    os.chdir(output)
    os.system('sh %s' % file)
    os.chdir(cwd)
    print('[dataset] Done!')


def download(output):
    if not os.path.exists(output):
        os.makedirs(output)
    tmp_dir = os.path.join(output, 'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    filename, cached_file = download_if_not_exists('data', tmp_dir, urls)
    unzip_file(cached_file, output)

    run_sh(os.path.join(output, 'raw_data_downloader.sh'), output)

    # image_folder = os.path.join(output, 'images')
    # if os.path.exists(image_folder):
    #     shutil.rmtree(image_folder)
    # os.mkdir(image_folder)
    #
    # for f in ['test2017', 'val2017', 'train2017']:
    #     folder = os.path.join(output, f)
    #     for i in os.listdir(folder):
    #         shutil.move(os.path.join(folder, i), image_folder)
    #     shutil.rmtree(folder)


if __name__ == '__main__':
    download('/Users/cenk.bircanoglu/wsl/wsl_survey/datasets/kitti')
    import argparse

    parser = argparse.ArgumentParser(description='Kitti Dataset Downloader')
    parser.add_argument('--dataset_dir',
                        metavar='DIR',
                        help='path to data dir')
    args = parser.parse_args()
    download(args.dataset_dir)
