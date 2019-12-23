import os
import os.path
import zipfile
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch
from tqdm import tqdm


def dict_to_file(items, filename, header=None, item_extractor=None):
    with open(filename, mode='w') as f:
        if header:
            f.write(header)
        for i in items:
            f.write(item_extractor(i))


def unzip_file(file, output):
    print('[dataset] Extracting zip file {file} to {path}'.format(file=file,
                                                                  path=output))
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(output)
    print('[dataset] Done!')


def make_one_hot(labels, C=2):
    one_hot = torch.zeros(C)
    one_hot[int(labels)] = 1
    return one_hot


def generate_class_id(trainval_split, tr_split, val_split, test_split):
    class_name_set = set()

    for item in trainval_split:
        class_name_set.add(item['class_name'])
    for item in tr_split:
        class_name_set.add(item['class_name'])
    for item in val_split:
        class_name_set.add(item['class_name'])
    for item in test_split:
        class_name_set.add(item['class_name'])
    class_name_id_map = {j: i for i, j in enumerate(sorted(class_name_set))}
    for item in trainval_split:
        item['class_id'] = class_name_id_map.get(item['class_name'])
    for item in tr_split:
        item['class_id'] = class_name_id_map.get(item['class_name'])
    for item in val_split:
        item['class_id'] = class_name_id_map.get(item['class_name'])
    for item in test_split:
        item['class_id'] = class_name_id_map.get(item['class_name'])
    mapping = [{
        'class_id': i[1],
        'class_name': i[0]
    } for i in class_name_id_map.items()]

    return trainval_split, tr_split, val_split, test_split, mapping


def download_if_not_exists(url_key, tmp_dir, urls):
    url = urls[url_key]
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(tmp_dir, filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url(url, cached_file)
    return filename, cached_file


def download_url(url, destination=None, progress_bar=True):
    """Download a URL to a local file.

    Parameters
    ----------
    url : str
        The URL to download.
    destination : str, None
        The destination of the file. If None is given the file is saved to a temporary directory.
    progress_bar : bool
        Whether to show a command-line progress bar while downloading.

    Returns
    -------
    filename : str
        The location of the downloaded file.

    Notes
    -----
    Progress bar use/example adapted from tqdm documentation: https://github.com/tqdm/tqdm
    """
    def progressbar_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(unit='B',
                  unit_scale=True,
                  miniters=1,
                  desc=url.split('/')[-1]) as t:
            filename, _ = urlretrieve(url,
                                      filename=destination,
                                      reporthook=progressbar_hook(t))
    else:
        filename, _ = urlretrieve(url, filename=destination)
