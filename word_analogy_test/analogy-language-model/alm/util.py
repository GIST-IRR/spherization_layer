""" Download files from web """
import tarfile
import zipfile
import requests
import gzip
import os
import shutil

import gdown
import random
import numpy as np
import torch

__all__ = ('open_compressed_file', 'wget', 'fix_seed')


def fix_seed(seed: int = 12):
    """ Fix random seed. """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def open_compressed_file(url, cache_dir, filename: str = None, gdrive: bool = False):
    path = wget(url, cache_dir, gdrive=gdrive, filename=filename)
    if path.endswith('.tar.gz') or path.endswith('.tgz'):
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(cache_dir)
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
    elif path.endswith('.gz'):
        with gzip.open(path, 'rb') as f_in:
            with open(path.replace('.gz', ''), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def wget(url, cache_dir, gdrive: bool = False, filename: str = None):
    os.makedirs(cache_dir, exist_ok=True)
    if gdrive:
        if filename:
            path = '{}/{}'.format(cache_dir, filename)
            if os.path.exists(path):
                return path

            return gdown.download(url, path, quiet=False)
        else:
            return gdown.download(url, cache_dir, quiet=False)
    filename = os.path.basename(url)
    path = '{}/{}'.format(cache_dir, filename)
    if os.path.exists(path):
        return path
    with open(path, "wb") as f:
        r = requests.get(url)
        f.write(r.content)
    return path