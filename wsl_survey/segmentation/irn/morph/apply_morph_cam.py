import numpy as np
import os
from functools import partial
from multiprocessing.pool import Pool
from numpy import newaxis
from scipy import ndimage
from skimage.morphology import erosion, opening, closing, dilation
from skimage.morphology import square
from tqdm import tqdm


def apply_morphology(imgs):
    selem = square(kernel_size)
    erodeds, dilateds, openeds, closeds, gaussians = [], [], [], [], []
    for img in imgs:
        erodeds.append(erosion(img * 255, selem)[newaxis, ...] / 255)
        dilateds.append(dilation(img * 255, selem)[newaxis, ...] / 255)
        openeds.append(opening(img * 255, selem)[newaxis, ...] / 255)
        closeds.append(closing(img * 255, selem)[newaxis, ...] / 255)
        gaussians.append(
            ndimage.gaussian_filter(img * 255, sigma=(kernel_size, kernel_size), order=0)[newaxis, ...] / 255)
    return np.concatenate(erodeds, axis=0), np.concatenate(dilateds,
                                                           axis=0), np.concatenate(
        openeds, axis=0), np.concatenate(closeds, axis=0), np.concatenate(
        gaussians, axis=0)


def create_morph(img_name, folder):
    eroded_folder = folder + '_eroded'
    dilated_folder = folder + '_dilated'
    opened_folder = folder + '_opened'
    closed_folder = folder + '_closed'
    gaussian_folder = folder + '_gaussian'

    cam_dict = np.load(os.path.join(folder, img_name),
                       allow_pickle=True).item()

    eroded, dilated, opened, closed, gaussians = apply_morphology(cam_dict['high_res'])
    eroded1, dilated1, opened1, closed1, gaussians1 = apply_morphology(cam_dict['cam'])
    assert cam_dict[
               'high_res'].shape == eroded.shape == dilated.shape == opened.shape == closed.shape == gaussians.shape

    cam_dict['high_res'] = eroded
    cam_dict['cam'] = eroded1
    np.save(os.path.join(eroded_folder, img_name), cam_dict)
    cam_dict['high_res'] = dilated
    cam_dict['cam'] = dilated1
    np.save(os.path.join(dilated_folder, img_name), cam_dict)
    cam_dict['high_res'] = opened
    cam_dict['cam'] = opened1
    np.save(os.path.join(opened_folder, img_name), cam_dict)
    cam_dict['high_res'] = closed
    cam_dict['cam'] = closed1
    np.save(os.path.join(closed_folder, img_name), cam_dict)
    cam_dict['high_res'] = gaussians
    cam_dict['cam'] = gaussians1
    np.save(os.path.join(gaussian_folder, img_name), cam_dict)
    return True


def apply(folder):
    grayscaled_folder = folder + '_grayscaled'
    eroded_folder = folder + '_eroded'
    dilated_folder = folder + '_dilated'
    opened_folder = folder + '_opened'
    closed_folder = folder + '_closed'
    gaussian_folder = folder + '_gaussian'

    os.makedirs(grayscaled_folder, exist_ok=True)
    os.makedirs(eroded_folder, exist_ok=True)
    os.makedirs(dilated_folder, exist_ok=True)
    os.makedirs(opened_folder, exist_ok=True)
    os.makedirs(closed_folder, exist_ok=True)
    os.makedirs(gaussian_folder, exist_ok=True)

    paths = os.listdir(folder)
    create_morph(paths[0], folder=folder)
    with Pool(processes=16) as pool:
        with tqdm(total=len(paths)) as pbar:
            for i, _ in tqdm(
                enumerate(
                    pool.imap_unordered(partial(create_morph, folder=folder),
                                        paths))):
                pbar.update()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--kernel_size", type=int, default=5)
    args = parser.parse_args()
    kernel_size = args.kernel_size
    apply('./outputs/voc12/results/resnet152/cam')
    apply('./outputs/voc12/results/resnet152/cam_val')
    # apply('/Users/cenk.bircanoglu/wsl/wsl_survey/results/resnet101/cam')
    # apply('/Users/cenk.bircanoglu/wsl/wsl_survey/results/resnet101/cam_val')
    # apply('/Users/cenk.bircanoglu/wsl/wsl_survey/results/resnet154/cam')
    # apply('/Users/cenk.bircanoglu/wsl/wsl_survey/results/resnet154/cam_val')
