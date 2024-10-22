#DB_optimize
import numpy as np
from bounds_utility import set_pbounds
from optimize import optimize_algorithm
from output_utility import output_data_sorter

import numpy.typing as npt
from typing import Tuple, Dict, Optional, Any, Union

def DB_Optimization(
    X: npt.NDArray[np.float_],
    runs: int,
    rand_n: int,
    algorithm: str, 
    mem_cutoff: float, 
    scale_params: bool = False, 
    **kwargs
) -> Tuple[npt.NDArray[Any], ]:  

    """
    Handles user inputs to pass to the Bayesian optimizer and formats
    the optimization outputs.

    Args:
        X (npt.NDArray[np.float_]):
            Array of data to be clustered with each column
            corresponding to a separate dimension. 

        runs (int):
            Sets the number of optimization iterations to optimize the 
            parameter space.

        rand_n (int):
            Sets the number of initial parameter combinations to probe before
            optimizing.

        algorithm (str):
            Currently capable of being set to 'DBSCAN', 'HDBSCAN', 
            or 'OPTICS'. See the sklearn clusering documentation
            for more information on each of these algorithms.

        scale_params (bool):
            Set to True will scale all parameter ranges equally 
            when optimizing. Default is set to False.

        mem_cutoff (float):
            Threshold for approximate memory usage to avoid high memory 
            allocation. Default is set to 25.0 GB, should be increassed or
            decreased dependent on a users system.

    Return:
        score_sweep_array ():
        
        median_ind_clust_scores ():

    
    """

    pbounds, constant_params, scale_parameter = set_pbounds(
        algorithm, scale_params, kwargs
        )


    optimizer, median_ind_clust_scores = optimize_algorithm(
        X, algorithm, pbounds, constant_params, runs, rand_n, scale_parameter, mem_cutoff,
        )

    score_sweep_array = output_data_sorter(
        optimizer, pbounds, scale_parameter
        )

    return score_sweep_array, median_ind_clust_scores