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

from .preprocessing import read_one_sample


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
        os.rmtree(path + name, ignore_errors=True)
        os.makedirs(name)
    else:
        pass


##


def set_logger(path_runs, name, mode='w'):
    """
    A function to open a logs.txt file for a certain script, writing its trace at path_main/runs/step/.
    """
    logger = logging.getLogger("mito_benchmark")
    handler = logging.FileHandler(path_runs + name, mode=mode)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger


##


def summary_stats_vars(afm, variants=None):
    """
    Calculate the most important summary stats for a bunch of variants, collected for
    a set of cells.
    """
    if variants is not None:
        test = afm.var_names.isin(variants)
        density = (~np.isnan(afm[:, test].X)).sum(axis=0) / afm.shape[0]
        median_vafs = np.nanmedian(afm[:, test].X, axis=0)
        median_coverage_var = np.nanmedian(afm[:, test].layers['coverage'], axis=0)
        fr_positives = np.sum(afm[:, test].X > 0, axis=0) / afm.shape[0]
        var_names = afm.var_names[test]
    else:
        density = (~np.isnan(afm.X)).sum(axis=0) / afm.shape[0]
        median_vafs = np.nanmedian(afm.X, axis=0)
        median_coverage_var = np.nanmedian(afm.layers['coverage'], axis=0)
        fr_positives = np.sum(afm.X > 0, axis=0) / afm.shape[0]
        var_names = afm.var_names

    df = pd.DataFrame(
        {   
            'density' : density,
            'median_coverage' : median_coverage_var,
            'median_AF' : median_vafs,
            'fr_positives' : fr_positives
        }, index=var_names
    )

    return df


##


def prep_things_for_umap(top_runs_per_sample, i, solutions, connectivities, path_main=None):
    """
    Utility used in leiden performance viz.
    """
    # Get top solutions
    d_run = top_runs_per_sample.iloc[i, :].to_dict()

    # Prepare ingredients for embs calculations
    s = d_run['sample']
    a = '_'.join(d_run['analysis'].split('_')[1:])

    path_ = path_main + f'results_and_plots/classification_performance/top_3/{s}/{a}/cell_x_var_hclust.pickle'

    with open(path_, 'rb') as f:
        d_cell_x_var = pickle.load(f)

    cells = d_cell_x_var['cells']
    variants = d_cell_x_var['vars']

    afm = read_one_sample(path_main, sample=s)
    X = afm[cells, variants].X.copy()

    conn_name = f'{d_run["analysis"]}_{d_run["with_nans"]}_{d_run["metric"]}_None'
    leiden_pickle_name = f'{d_run["analysis"]}_{d_run["with_nans"]}_{d_run["metric"]}_None|{d_run["k"]}|{d_run["res"]}'

    labels, true_clones, ARI = solutions[s][leiden_pickle_name]
    conn = connectivities[s][conn_name]

    return X, conn, cells, true_clones, labels, ARI, d_run


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