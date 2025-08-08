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

def Est_Data(folder: Union[str, PosixPath], DataFileName: str, N: int,
             indices: np.ndarray=None, T: float = 5400, mode: str = 'sim',
             method: str = 'GLM', WIN:float=50.0, DELTA:float=1.0,
             outfile_sfx:str=None):
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
    start_cpu_time = time.process_time()
    start_wall_time = time.time()

    cc_time = 0.0
    for i in range(0, N):
        for j in range(0, i):
            X1 = input_data[input_data[:, 1] == i, 0]
            X2 = input_data[input_data[:, 1] == j, 0]

            print(f"Estimating PSPs from node {j} to {i}...")
            #Make cross_correlogram
            t0 = time.time()
            cc_list = linear_crossCorrelogram(X1, X2, T)
            t1 = time.time()
            cc_time += t1 - t0

            #set tau
            tau = [4, 4]

            #Fitting a GLM
            if mode == 'sim':
                delay_synapse = 3
                par, log_pos, log_likelihood = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse)
            elif mode == 'exp':
                log_pos = 0
                log_likelihood = 0
                for m in range(1, 5):
                    tmp_par, tmp_log_pos, tmp_log_likelihood = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], m)
                    if m == 1 or (not LR and tmp_log_pos > log_pos) or (LR and tmp_log_likelihood > log_likelihood):
                        log_pos = tmp_log_pos
                        log_likelihood = tmp_log_likelihood
                        par = tmp_par
                        delay_synapse = m
            else:
                raise ValueError("Input error: You must write sim or exp in mode")
                        

            #Connection parameters
            nb = int(WIN/DELTA)
            cc_0 = [0 for l in range(2)]
            max = [0 for l in range(2)]
            Jmin = [0 for l in range(2)]
            for l in range(2):
                cc_0[l] = 0
                max[l] = int(tau[l] + 0.1)
                
                if l == 0:
                    for m in range(max[l]):
                        cc_0[l] += np.exp(par[nb+int(delay_synapse)+m])
                if l == 1:
                    for m in range(max[l]):
                        cc_0[l] += np.exp(par[nb-int(delay_synapse)-m])

                cc_0[l] = cc_0[l]/max[l]
                        
                Jmin[l] = math.sqrt(16.3/ tau[l]/ cc_0[l])
                n12 = tau[l]*cc_0[l]
                if n12 <= 10:
                    par[NPAR-2+l] = 0
            D1 = 0
            D2 = 0
            if LR:
                tmp_par, tmp_log_pos, log_likelihood_p = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse, cond = 1)
                tmp_par, tmp_log_pos, log_likelihood_n = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse, cond = 2)
                D1 = log_likelihood - log_likelihood_p
                D2 = log_likelihood - log_likelihood_n

            #Output J
            # DataFileName.split
            J_f = open(buff, 'a')
            J_f.write(str(i)+' '+str(j)+' '
                    +str(round(par[NPAR-1], 6))+' '+str(round(par[NPAR-2], 6))+' '
                    +str(round(Jmin[1], 6))+' '+str(round(Jmin[0], 6))+' '
                    +str(round(D2, 6))+' '+str(round(D1, 6))+'\n')
            J_f.close()

    scale = 1.277
    z_a = 15.14

    #Read the required J file and create the resul file
    J_f = open(buff, 'r')
    J_f_list = J_f.readlines()
    W = np.zeros((N, N))

    #calculate W
    for i in range(0, len(J_f_list)):
        J_f_list[i] = J_f_list[i].split()
        
        J_f_list[i][0] = int(float(J_f_list[i][0])) #pre
        J_f_list[i][1] = int(float(J_f_list[i][1])) #post
        J_f_list[i][2] = float(J_f_list[i][2])      #J_+
        J_f_list[i][3] = float(J_f_list[i][3])      #J_-
        J_f_list[i][4] = float(J_f_list[i][4])      #J_min_+
        J_f_list[i][5] = float(J_f_list[i][5])      #J_min_-
        J_f_list[i][6] = float(J_f_list[i][6])      #D_+
        J_f_list[i][7] = float(J_f_list[i][7])      #D_-
        
        if not LR:
            W[J_f_list[i][0], J_f_list[i][1]] = round(calc_PSP(J_f_list[i][2], J_f_list[i][4]*scale), 6)
            W[J_f_list[i][1], J_f_list[i][0]] = round(calc_PSP(J_f_list[i][3], J_f_list[i][5]*scale), 6)
        else:
            W[J_f_list[i][0], J_f_list[i][1]] = round(calc_PSP_LR(J_f_list[i][2], J_f_list[i][6], z_a), 6)
            W[J_f_list[i][1], J_f_list[i][0]] = round(calc_PSP_LR(J_f_list[i][3], J_f_list[i][7], z_a), 6)

    #write W
    if outfile_sfx is not None:
        output_file = folder / f"W_{method:s}_{T:.0f}-{DataFileName:s}_{outfile_sfx}.npy"
    else:
        output_file = folder / f"W_{method:s}_{T:.0f}-{DataFileName:s}.npy"
    np.save(output_file, W)

    #remove J file

    # debug
    # cmd = ['rm', str(folder/f"J_py_{T:.0f}.txt")]
    # proc.check_call(cmd)

    # End measuring CPU time and wall time
    end_cpu_time = time.process_time()
    end_wall_time = time.time()

    # Calculate the elapsed CPU time and wall time
    elapsed_cpu_time = end_cpu_time - start_cpu_time
    elapsed_wall_time = end_wall_time - start_wall_time

    # Print the elapsed CPU time and wall time
    print(f"CC estimation time: {cc_time:.3f} s")
    print(f"Elapsed CPU time: {elapsed_cpu_time:.3f} s")
    print(f"Elapsed Wall time: {elapsed_wall_time:.3f} s")
    return W, elapsed_cpu_time, elapsed_wall_time
