""""
Miscellaneous utilities.
"""

from shutil import rmtree
import logging
import os 
import time 
import pickle
import numpy as np
import pandas as pd
import scanpy as sc


##


class TimerError(Exception):
    """
    A custom exception used to report errors in use of Timer class.
    """

class Timer:
    """
    A custom Timer class.
    """
    def __init__(self):
        self._start_time = None

    def start(self):
        """
        Start a new timer.
        """
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def stop(self):
        """
        Stop the timer, and report the elapsed time.
        """
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        elapsed_time = time.perf_counter() - self._start_time

        if elapsed_time > 100:
            unit = 'min'
            elapsed_time = elapsed_time / 60
        elif elapsed_time > 1000:
            unit = 'h'
            elapsed_time = elapsed_time / 3600
        else:
            unit = 's'

        self._start_time = None

        return f'{round(elapsed_time, 2)} {unit}'


##


def make_folder(path, name, overwrite=True):
    """
    A function to create a new {name} folder at the {path} path.
    """
    os.chdir(path)
    if not os.path.exists(name) or overwrite:
        rmtree(os.path.join(path, name), ignore_errors=True)
        os.makedirs(name)
    else:
        pass


##


def set_logger(path, name, mode='w'):
    """
    A function to open a logs.txt file for a certain script, writing its trace at path_main/runs/step/.
    """
    logger = logging.getLogger("mito_benchmark")
    handler = logging.FileHandler(os.path.join(path, name), mode=mode)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger


##


def update_params(d_original, d_passed):
    for k in d_passed:
        if k in d_original:
            pass
        else:
            print(f'{k}:{d_passed[k]} kwargs added...')
        d_original[k] = d_passed[k]
        
    return d_original


##


def one_hot_from_labels(y):
    """
    My one_hot encoder from a categorical variable.
    """
    if len(y.categories) > 2:
        Y = np.concatenate(
            [ np.where(y == x, 1, 0)[:, np.newaxis] for x in y.categories ],
            axis=1
        )
    else:
        Y = np.where(y == y.categories[0], 1, 0)
    
    return Y


##


def rescale(x):
    """
    Max/min rescaling.
    """    
    if np.min(x) != np.max(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    else:
        return x
    

##


def ji(x, y):
    """
    Jaccard Index between two list-like objs.
    """
    x = set(x)
    y = set(y)
    ji = len(x&y) / len(x|y)

    return ji


