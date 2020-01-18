import os
from functools import partial
from multiprocessing.pool import Pool

import imageio
from PIL import Image
from scipy import ndimage
from skimage.morphology import erosion, opening, closing, dilation
from skimage.morphology import square
from tqdm import tqdm


def apply_morphology(img):
    selem = square(kernel_size)
    return erosion(img,
                   selem), dilation(img, selem), opening(img, selem), closing(
                       img,
                       selem), ndimage.gaussian_filter(img,
                                                       sigma=(kernel_size,
                                                              kernel_size),
                                                       order=0)


def create_morph(img_name, folder):
    eroded_folder = folder + '_eroded'
    dilated_folder = folder + '_dilated'
    opened_folder = folder + '_opened'
    closed_folder = folder + '_closed'
    gaussian_folder = folder + '_gaussian'
    im = Image.open(os.path.join(folder, img_name))

    eroded, dilated, opened, closed, gaussians = apply_morphology(im)
    assert eroded.shape == dilated.shape == opened.shape == closed.shape == gaussians.shape

    imageio.imsave(os.path.join(eroded_folder, img_name), eroded)
    imageio.imsave(os.path.join(dilated_folder, img_name), dilated)
    imageio.imsave(os.path.join(opened_folder, img_name), opened)
    imageio.imsave(os.path.join(closed_folder, img_name), closed)
    imageio.imsave(os.path.join(gaussian_folder, img_name), gaussians)
    return True


def apply(folder):
    eroded_folder = folder + '_eroded'
    dilated_folder = folder + '_dilated'
    opened_folder = folder + '_opened'
    closed_folder = folder + '_closed'
    gaussian_folder = folder + '_gaussian'

    os.makedirs(eroded_folder, exist_ok=True)
    os.makedirs(dilated_folder, exist_ok=True)
    os.makedirs(opened_folder, exist_ok=True)
    os.makedirs(closed_folder, exist_ok=True)
    os.makedirs(gaussian_folder, exist_ok=True)

    paths = os.listdir(folder)
    with Pool(processes=4) as pool:
        with tqdm(total=len(paths)) as pbar:
            for i, _ in tqdm(
                    enumerate(
                        pool.imap_unordered(
                            partial(create_morph, folder=folder), paths))):
                pbar.update()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--kernel_size", default=3, type=int)
    args = parser.parse_args()
    kernel_size = args.kernel_size
    apply('./outputs/voc12/results/$MODEL/irn_label')
    apply('./outputs/voc12/results/$MODEL/irn_label_val')
    # apply('/Users/cenk.bircanoglu/wsl/wsl_survey/results/resnet101/cam')
    # apply('/Users/cenk.bircanoglu/wsl/wsl_survey/results/resnet101/cam_val')
    # apply('/Users/cenk.bircanoglu/wsl/wsl_survey/results/resnet154/cam')
    # apply('/Users/cenk.bircanoglu/wsl/wsl_survey/results/resnet154/cam_val')
