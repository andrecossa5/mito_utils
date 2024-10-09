""""
Miscellaneous utilities.
"""

import os 
import sys
import re
import time 
import random
import string
import pickle
import json
from shutil import rmtree
import logging
import numpy as np
import pandas as pd
import scanpy as sc


##


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'  # Custom format
)


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


##


def process_char_filtering_kwargs(path_filtering, filtering_key):
    """
    Processing the filtering options .json file.
    """
    with open(path_filtering, 'r') as file:
        FILTERING_OPTIONS = json.load(file)
    
    if filtering_key in FILTERING_OPTIONS:
        d = FILTERING_OPTIONS[filtering_key]
        filtering = d['filtering']
        filtering_kwargs = d['filtering_kwargs'] if 'filtering_kwargs' in d else {}
        kwargs = { k : d[k] for k in d if k not in ['filtering', 'filtering_kwargs'] }
        kwargs = {k: True if v == "True" else v for k, v in kwargs.items()}         # Nextflow and .json pain
        kwargs = {k: False if v == "False" else v for k, v in kwargs.items()}
    else:
        raise KeyError(f'{filtering_key} not in {path_filtering}!')
    
    return filtering, filtering_kwargs, kwargs


##


def process_bin_kwargs(path_bin, bin_key):
    """
    Processing the filtering options .json file.
    """
    
    with open(path_bin, 'r') as file:
        BIN_OPTIONS = json.load(file)
    
    if bin_key in BIN_OPTIONS:
        d = BIN_OPTIONS[bin_key]
        bin_method = d['bin_method']
        binarization_kwargs = d['binarization_kwargs'] if 'binarization_kwargs' in d else {}
    else:
        raise KeyError(f'{bin_key} not in {path_bin}!')
    
    return bin_method, binarization_kwargs


##


def process_kwargs(path, key):
    """
    Processing .json file.
    """

    with open(path, 'r') as file:
        OPTIONS = json.load(file)
    kwargs = OPTIONS[key] if key in OPTIONS else {}

    return kwargs


##


def traverse_and_extract_flat(base_dir, file_name='annotated_tree.pickle'):

    result = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == file_name:
                full_path = os.path.join(root, file)
                
                # Determine the file extension and load accordingly
                if file_name.endswith('.pickle'):
                    with open(full_path, 'rb') as f:
                        data = pickle.load(f)
                elif file_name.endswith('.csv'):
                    data = pd.read_csv(full_path, index_col=0)
                elif file_name.endswith('.txt.gz'):
                    data = pd.read_csv(full_path, index_col=0, header=None)
                else:
                    raise ValueError(f"Unsupported file type: {file_name}")

                # Extracting the relevant folder parts
                relative_path = os.path.relpath(root, base_dir)
                folder_parts = tuple(relative_path.split(os.sep))
                
                # Using tuple of folder parts as the key
                result[folder_parts] = data

    return result


##


def generate_job_id(id_length=20):

    characters = string.ascii_letters + string.digits  # All uppercase, lowercase letters, and digits
    random_id = ''.join(random.choices(characters, k=id_length))
    random_id = f'job_{random_id}'

    return random_id


##


def get_metrics(path, metric_pattern=None):
    """
    Retrieve metrics df.
    """
    L = []
    for folder,_,files in os.walk(path):
        for file in files:
            if bool(re.search(metric_pattern, file)): 
                sample = folder.split('/')[-2]
                job_id = folder.split('/')[-1]
                df = pd.read_csv(os.path.join(folder,file), index_col=0).assign(sample=sample)
                if 'job_id' not in df.columns:
                    df = df.assign(job_id=job_id)
                df = df.rename(columns={'value':'metric_value'})
                L.append(df)
    df_metrics = pd.concat(L)

    L = []
    for folder,_,files in os.walk(path):
        for file in files:
            if bool(re.search('ops', file)): 
                df = pd.read_csv(os.path.join(folder,file), index_col=0)
                df = df.rename(columns={'value':'option_value'})
                L.append(df)
    df_ops = pd.concat(L).pivot(index='job_id', columns='option', values='option_value').reset_index()

    df = df_metrics.merge(df_ops, on='job_id')

    return df


##