#Author: Junichi haruna
'''
Estimate PSPs from Data.

Input : Data file and the number of data

'''

from .glmcc import *
import time
import tempfile
import numpy as np
import math
from pathlib import Path, PosixPath
from typing import Union
from multiprocessing import Pool
SCALE = 1.277
Z_A = 15.14

def _process_pair(X1: np.ndarray, X2: np.ndarray, T:float, mode: str, LR:bool, beta:float, WIN:float, DELTA:float) -> float:
    # Make cross-correlogram
    t0 = time.process_time()
    print('start computing correlation ...')
    cc_list = linear_crossCorrelogram(X1, X2, T)

    # set tau
    tau = [4, 4]

    # Fitting a GLM
    if mode == 'sim':
        delay_synapse = 3
        par, log_pos, log_likelihood = GLMCC(
            cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse
        )
    elif mode == 'exp':
        log_pos = 0
        log_likelihood = 0
        for m in range(1, 5):
            tmp_par, tmp_log_pos, tmp_log_likelihood = GLMCC(
                cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], m
            )
            if m == 1 or (not LR and tmp_log_pos > log_pos) or (LR and tmp_log_likelihood > log_likelihood):
                log_pos = tmp_log_pos
                log_likelihood = tmp_log_likelihood
                par = tmp_par
                delay_synapse = m
    else:
        raise ValueError("Input error: You must write sim or exp in mode")

    # Connection parameters
    nb = int(WIN / DELTA)
    cc_0 = [0 for _ in range(2)]
    max_l = [0 for _ in range(2)]
    Jmin = [0 for _ in range(2)]
    for l in range(2):
        cc_0[l] = 0
        max_l[l] = int(tau[l] + 0.1)

        if l == 0:
            for m in range(max_l[l]):
                cc_0[l] += np.exp(par[nb + int(delay_synapse) + m])
        if l == 1:
            for m in range(max_l[l]):
                cc_0[l] += np.exp(par[nb - int(delay_synapse) - m])

        cc_0[l] = cc_0[l] / max_l[l]

        Jmin[l] = math.sqrt(16.3 / tau[l] / cc_0[l])
        n12 = tau[l] * cc_0[l]
        if n12 <= 10:
            par[NPAR - 2 + l] = 0
    # calculate W
    if not LR:
        Wij = round(calc_PSP(par[NPAR - 1], Jmin[1]*SCALE), 6)
        Wji = round(calc_PSP(par[NPAR - 2], Jmin[0]*SCALE), 6)
    else:
        D1 = 0
        D2 = 0
        _, _, log_likelihood_p = GLMCC(
            cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse, cond=1
        )
        _, _, log_likelihood_n = GLMCC(
            cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse, cond=2
        )
        D1 = log_likelihood - log_likelihood_p
        D2 = log_likelihood - log_likelihood_n
        Wij = round(calc_PSP_LR(par[NPAR - 1], D2, Z_A), 6)
        Wji = round(calc_PSP_LR(par[NPAR - 2], D1, Z_A), 6)

    #pre : i
    #post : j
    #J_+  : par[NPAR - 1]
    #J_-  : par[NPAR - 2]
    #J_min_+ : Jmin[1]
    #J_min_- : Jmin[0]
    #D_+    : D2
    #D_-    : D1
    
    t1 = time.process_time()
    elapsed_time = t1 - t0

    return Wij, Wji, elapsed_time


def Est_Data(folder: Union[str, PosixPath], DataFileName: str, N: int,
             indices: np.ndarray=None, T: float = 5400, mode: str = 'sim',
             method: str = 'GLM', WIN:float=50.0, DELTA:float=1.0,
             outfile_sfx:str=None, n_jobs:int=1):
    """
    Estimate synaptic connection strengths (PSPs) between neurons from spike data using GLM or LR methods.
    This function loads spike data, computes cross-correlograms for all neuron pairs, fits a Generalized Linear Model (GLM) or Logistic Regression (LR) to estimate synaptic parameters, and outputs a connectivity matrix as a CSV file.

    Parameters
    ----------
    folder : Union[str, PosixPath]
        Path to the folder containing the data file.
    DataFileName : str
        Name of the .npy file containing spike data (shape: (n_spikes, 2), columns: spike time, cell index).
    N : int
        Number of neurons in the dataset.
    indices : np.ndarray, optional
        Indices of neurons to consider for analysis (default is None, which means all neurons are considered).
    T : float, optional
        Maximum time (in seconds) to consider for analysis (default is 5400).
    mode : str, optional
        Mode of operation: 'sim' for simulated data or 'exp' for experimental data (default is 'sim').
    method : str, optional
        Estimation method: 'GLM' for Generalized Linear Model or 'LR' for Logistic Regression (default is 'GLM').
    WIN : float, optional
        Window size for cross-correlogram (default is 50.0).
    DELTA : float, optional
        Bin size for cross-correlogram (default is 1.0).
    outfile_sfx : str, optional
        Suffix for the output file name (default is None, which means no suffix is added).
    n_jobs : int, optional
        number of parallel workers
        
    Returns
    -------
    None
        Writes the estimated connectivity matrix to a CSV file in the specified folder.

    Notes
    -----
    - The function prints timing information for cross-correlogram estimation and total execution.
    - The output CSV file is named as "W_{method}_{T}-{DataFileName}.csv".
    - Requires external functions: `linear_crossCorrelogram`, `GLMCC`, `calc_PSP`, and `calc_PSP_LR`.
    - Temporary files are used for intermediate results and are removed after processing.
    """

    # import sys

    folder = Path(folder)
    if method == "GLM":
        LR = False
        beta = 4000
    elif method == "LR":
        LR = True
        beta = 10000
    else:
        raise ValueError("Method must be 'GLM' or 'LR'.")

    buff = tempfile.mktemp()

    input_data = np.fromfile(
        folder/(DataFileName+'_spike_train.dat'), dtype=float).reshape(-1,2)   # shape (n_spikes, 2), first column is spike time, second column is cell index
    input_data = input_data[input_data[:, 0] <= T*1e3]
    if indices is not None:
        N = len(indices)
        mask = None
        for id_ in indices:
            if mask is None:
                mask = input_data[:, 1] == id_
            else:
                mask |= input_data[:, 1] == id_
        input_data = input_data[mask]
        for new_id, old_id in enumerate(indices):
            input_data[input_data[:, 1] == old_id, 1] = new_id

    # Start measuring CPU time and wall time
    start_wall_time = time.time()

    index_pairs = [(i,j) for i in range(N) for j in range(i)]
    pool = Pool(n_jobs)
    results = [pool.apply_async(
        _process_pair, args=(input_data[input_data[:, 1] == i, 0], input_data[input_data[:, 1] == j, 0], T, mode, LR, beta, WIN, DELTA)
    ) for i, j in index_pairs]
    pool.close()
    pool.join()
    # results_list = []
    # for i, j in index_pairs:
    #     # Helper function to process a neuron pair (i -> j)
    #     X1 = input_data[input_data[:, 1] == i, 0]
    #     X2 = input_data[input_data[:, 1] == j, 0]
    #     print(f"Estimating PSPs from node {i} to {j}...")
    #     results = _process_pair(X1, X2)
    #     results_list.append((i, j, *results))


    #Read the required J file and create the resul file
    W = np.zeros((N, N))
    cpu_time = 0.0
    for (i, j), res in zip(index_pairs, results):
        W[i, j] = res.get()[0]
        W[j, i] = res.get()[1]
        cpu_time += res.get()[2]

    #write W
    if outfile_sfx is not None:
        output_file = folder / f"W_{method:s}_{T:.0f}-{DataFileName:s}_{outfile_sfx}.npy"
    else:
        output_file = folder / f"W_{method:s}_{T:.0f}-{DataFileName:s}.npy"
    np.save(output_file, W)

    # End measuring CPU time and wall time
    end_wall_time = time.time()

    # Calculate the elapsed CPU time and wall time
    elapsed_wall_time = end_wall_time - start_wall_time

    # Print the elapsed CPU time and wall time
    print(f"Elapsed CPU time: {cpu_time:.3f} s")
    print(f"Elapsed Wall time: {elapsed_wall_time:.3f} s")
    return W, cpu_time, elapsed_wall_time
